"""Qwen3-30B-A3B 12h training — curated bugfix+Vulkan dataset with USAF sparse fine-tuning."""
import json, math, os, random, time
from pathlib import Path
import psutil, torch
from transformers import AutoConfig
from safetensors import safe_open
from usaf.qwen3moe_dml import patch_qwen3moe_for_dml
from usaf.sparse_optim import SparseAdam
from usaf.utils import get_dml_device
from usaf.moe_loader import QuantizedExpertCache, SparseGradStore, TopKImportanceStore

os.environ["TQDM_DISABLE"]="1"
proc=psutil.Process(os.getpid())
def ram(): return proc.memory_info().rss/1024**3

# ── 12h Config ──
SRC="Qwen3-30B-A3B"; Q4="Qwen3-30B-A3B-q4"
SEQ=512
FRAC=float(os.environ.get("FRAC","5e-3"))         # fraction of active elements per tensor
STEPS=int(os.environ.get("STEPS","180"))
EPOCHS=float(os.environ.get("EPOCHS","0"))          # if >0, recalculate STEPS from epochs
MICROBATCH=int(os.environ.get("MICROBATCH","4"))   # real batch size (streaming, single pass)
ACCUM=int(os.environ.get("ACCUM","1"))             # accumulation micro-batches; effective batch = MICROBATCH*ACCUM
# smoke overrides (produção inalterada com defaults)
STEPS=int(os.environ.get("SMOKE_STEPS",STEPS)); _SMOKE_N=int(os.environ.get("SMOKE_N","0"))
LR_PEAK=float(os.environ.get("LR_PEAK","2e-4")); WD=0.005
_L0=int(os.environ.get("TRAIN_FROM","40"))         # capacidade: primeira camada treinavel
TRAIN_LAYERS=set(range(_L0,48))
EVAL_EVERY=int(os.environ.get("SMOKE_EVAL",int(os.environ.get("EVAL_EVERY","15"))))
N_IMPORTANCE=int(os.environ.get("SMOKE_IMP",3))
LOSS_SCALE_INIT=4096.0; SCALE_UP_EVERY=200
RESELECT_EVERY=int(os.environ.get("SMOKE_RESELECT",os.environ.get("RESELECT_EVERY","50")))
RESELECT_DROP=float(os.environ.get("RESELECT_DROP","0.1"))
_TAG="_smoke" if _SMOKE_N else os.environ.get("RUN_TAG","")
CKPT_PATH=f"checkpoints/qwen3_12h_ckpt{_TAG}.pt"
CKPT_EVERY_SEC=900
LOG_PATH=f"logs/qwen3_12h{_TAG}.jsonl"
DATASET_PATH="data/train_dataset_12h.jsonl"
USE_FROZEN_CACHE=os.environ.get("USE_FROZEN_CACHE","1")=="1"
FROZEN_CACHE_N=int(os.environ.get("FROZEN_CACHE_N","0"))  # 0=all, N=cache first N samples
USE_VK=os.environ.get("USE_VK","0")=="1"                      # Vulkan attention forward
USE_VK_DEQUANT=os.environ.get("USE_VK_DEQUANT","0")=="1"      # Vulkan GPU dequant (default OFF: too slow)
USE_CUDA=os.environ.get("USE_CUDA","0")=="1"                  # CUDA backend (NVIDIA)
USE_AMP=os.environ.get("USE_AMP","1")=="1"                    # Automatic Mixed Precision
USE_MULTI_GPU=os.environ.get("USE_MULTI_GPU","1")=="1"        # DataParallel multi-GPU
FROZEN_CACHE_PATH=f"checkpoints/frozen_cache_12h{_TAG}_d{min(TRAIN_LAYERS)-1}.npy"
FROZEN_CACHE=None                        # set after importance selection
VK_LAYERS={}                             # layer_idx -> VKLayer
DETACH_AT=min(TRAIN_LAYERS)-1
assert TRAIN_LAYERS==set(range(min(TRAIN_LAYERS),max(TRAIN_LAYERS)+1)),"TRAIN_LAYERS must be contiguous"

# ── 1. Dataset (pre-tokenized JSONL) ──
print("="*60); print("1/4  Dataset (12h curated)")
print("="*60)
random.seed(42)

def load_jsonl(path):
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

samples=load_jsonl(DATASET_PATH)
random.shuffle(samples)
sp=max(1,int(len(samples)*0.97))
train_samples=samples[:sp]; eval_samples=samples[sp:sp+10]
if _SMOKE_N:
    train_samples=train_samples[:_SMOKE_N]; eval_samples=eval_samples[:2]
# frozen cache indices (train + eval + heldout)
_sidx=0
for _i,_s in enumerate(train_samples): _s["_fidx"]=_sidx; _sidx+=1
for _i,_s in enumerate(eval_samples): _s["_fidx"]=_sidx; _sidx+=1
# held-out eval: repositories outside training (flecs/sfml/entt/box2d)
heldout_samples=load_jsonl("data/eval_heldout_12h.jsonl")
random.shuffle(heldout_samples)
if _SMOKE_N: heldout_samples=heldout_samples[:2]
for _i,_s in enumerate(heldout_samples): _s["_fidx"]=_sidx; _sidx+=1
print(f"  {len(train_samples)} train + {len(eval_samples)} eval blocks of {SEQ} tokens "
      f"({len(train_samples)*SEQ:,} train tokens total)")
print(f"  {len(heldout_samples)} held-out blocks (repos fora do treino)")
_EFF=ACCUM*MICROBATCH
if EPOCHS>0 and not _SMOKE_N:
    _tokens_per_epoch=len(train_samples)*SEQ
    _desired_steps=max(1,int(EPOCHS*_tokens_per_epoch/(_EFF*SEQ)))
    print(f"  EPOCHS={EPOCHS:.1f}: recalculating STEPS {STEPS} -> {_desired_steps}")
    STEPS=_desired_steps
print(f"  Throughput: {STEPS} steps × {ACCUM} accum × {MICROBATCH} microbatch × {SEQ} seq = {STEPS*_EFF*SEQ:,} tokens")
print(f"  Epochs: ~{STEPS*_EFF*SEQ/(len(train_samples)*SEQ):.1f}x over {len(train_samples)*SEQ:,} unique tokens")

# ── 2. Model (fully manual, streaming quantized) ──
print(f"\n2/4  Model ({len(train_samples)} samples, {STEPS} steps)")
if USE_CUDA:
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print(f"  CUDA: {n_gpus} GPU(s) detected")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {p.name} ({p.total_mem/1e9:.1f}GB)")
    if USE_AMP:
        _amp_scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  AMP + cuDNN benchmark + TF32 enabled")
    else:
        _amp_scaler = None
else:
    patch_qwen3moe_for_dml()
    import torch_directml_native
    torch_directml_native.disable_tiled_resources(True)
    device = get_dml_device()
    n_gpus = 1
    _amp_scaler = None
cpu = torch.device("cpu")

with torch.device("meta"):
    cfg=AutoConfig.from_pretrained(SRC)
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
    model=Qwen3MoeForCausalLM(cfg)

st_files=sorted([f for f in os.listdir(SRC) if f.endswith(".safetensors")])
wf={}
for fn in st_files:
    with safe_open(os.path.join(SRC,fn),framework="pt") as sf:
        for key in sf.keys(): wf[key]=fn

mp=dict(model.named_parameters()); n=0; _gate_params={}
_w_np={}  # numpy weights for Vulkan (trainable layers only)
for name in sorted(wf.keys()):
    if ".mlp.experts." in name: continue
    if name not in mp: continue
    with safe_open(os.path.join(SRC,wf[name]),framework="pt") as sf:
        tensor=sf.get_tensor(name).half()
    parts=name.split("."); obj=model
    _layer = int(parts[2]) if len(parts)>2 and parts[0]=="model" and parts[1]=="layers" else -1
    if _layer in TRAIN_LAYERS:
        _w_np[name]=tensor.cpu().numpy()
    for p in parts[:-1]: obj=getattr(obj,p)
    _is_gate = ".mlp.gate.weight" in name
    _layer = int(parts[2]) if len(parts)>2 and parts[0]=="model" and parts[1]=="layers" else -1
    _train_gate = _is_gate and _layer in TRAIN_LAYERS
    gpu_tensor=tensor.to(device)
    obj._parameters[parts[-1]]=torch.nn.Parameter(gpu_tensor,requires_grad=_train_gate)
    if _train_gate: _gate_params[name]=obj._parameters[parts[-1]]
    n+=1
print(f"  {n} non-expert -> GPU")

for mn,mod in model.named_modules():
    for bn,b in list(mod._buffers.items()):
        if b is not None and b.device.type=="meta":
            if bn=="inv_freq":
                hd=getattr(mod,"dim",getattr(mod,"head_dim",128))
                base=getattr(mod,"base",1000000.0)
                inv=1.0/(base**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
                mod._buffers[bn]=inv.to(dtype=torch.float16,device=device)
            else:
                mod._buffers[bn]=torch.zeros(b.shape,dtype=torch.float16,device=device)

q_dict=torch.load(os.path.join(Q4,"experts_q4.pt"),map_location="cpu",weights_only=True)
cache=QuantizedExpertCache(q_dict,device,max_cached=1,group_size=128)

for mname,mod in model.named_modules():
    if mname.endswith(".mlp.experts"):
        mod._parameters.clear()
        if hasattr(mod,'_buffers'): mod._buffers.clear()

def _q_shape(fn):
    e=q_dict[fn]
    return tuple(e["shape"] if isinstance(e,dict) else e[3])
def _experts_name(i): return f"model.layers.{i}.mlp.experts"
_train_names=[f"model.layers.{li}.mlp.experts.{pn}"
              for li in sorted(TRAIN_LAYERS) for pn in ("gate_up_proj","down_proj")]
_shapes={fn:_q_shape(fn) for fn in _train_names}
capture_store=TopKImportanceStore(_shapes,frac=FRAC)

for mname,mod in model.named_modules():
    if not mname.endswith(".mlp.experts"): continue

    def make_pre(name):
        def pre(module,args):
            weights=cache.get_expert_weights(name)
            for pn,param in weights.items():
                module._parameters[pn]=param
        return pre

    def make_post():
        def post(module,args,output):
            module._parameters.clear()
            return output
        return post

    mod._grad_capture=(capture_store,mname)
    mod.register_forward_pre_hook(make_pre(mname))
    mod.register_forward_hook(make_post())

# ── Vulkan persistent forward for attention ──
if USE_VK:
    from usaf.vk_layer import create_vk_layers
    import numpy as np
    _vk_w = {}
    for li in sorted(TRAIN_LAYERS):
        prefix = f"model.layers.{li}."
        w = {k[len(prefix):]: v for k, v in _w_np.items() if k.startswith(prefix)}
        if w: _vk_w[li] = w
    if _vk_w:
        VK_LAYERS = create_vk_layers(TRAIN_LAYERS, model.config, _vk_w, model.model.rotary_emb)
        print(f"  Vulkan layers: {len(VK_LAYERS) if VK_LAYERS else 0} ready")
        _VK_HD = getattr(model.config, 'head_dim', 128)
        _VK_NH = model.config.num_attention_heads
        _VK_NKV = model.config.num_key_value_heads
    else:
        print(f"  Vulkan: no weights loaded")
else:
    VK_LAYERS = {}
    _VK_HD = _VK_NH = _VK_NKV = 0

model.train()
print(f"  Manual streaming: {sum(1 for n,m in model.named_modules() if n.endswith('.mlp.experts'))} expert modules")
if USE_CUDA and USE_MULTI_GPU and n_gpus > 1:
    model = torch.nn.DataParallel(model)
    print(f"  Multi-GPU: DataParallel across {n_gpus} GPUs")
print(f"  Layers: {sorted(TRAIN_LAYERS)} | SEQ={SEQ} | ACCUM={ACCUM} | STEPS={STEPS}")
print(f"  RAM: {ram():.1f}GB")

# ── 3. Forward/backward manual ──
loss_scale=LOSS_SCALE_INIT
N_LAYERS=len(model.model.layers)

def _prelude(input_ids):
    """Full prelude: embedding + RoPE + causal mask."""
    hidden=model.model.embed_tokens(input_ids)
    seq_len=hidden.shape[1]
    pos_ids=torch.arange(seq_len,device=device).unsqueeze(0)
    cos,sin=model.model.rotary_emb(hidden,position_ids=pos_ids)
    causal_mask=torch.triu(
        torch.full((seq_len,seq_len),torch.finfo(torch.float16).min,device=device,dtype=torch.float16),
        diagonal=1).unsqueeze(0).unsqueeze(0)
    return hidden,pos_ids,(cos,sin),causal_mask

def _prelude_lite(input_ids):
    """Prelude without embedding (for eval with frozen cache)."""
    B,S=input_ids.shape[0],input_ids.shape[1]
    pos_ids=torch.arange(S,device=device).unsqueeze(0)
    dummy=torch.zeros(B,S,model.config.hidden_size,dtype=torch.float16,device=device)
    cos,sin=model.model.rotary_emb(dummy,position_ids=pos_ids)
    causal_mask=torch.triu(
        torch.full((S,S),torch.finfo(torch.float16).min,device=device,dtype=torch.float16),
        diagonal=1).unsqueeze(0).unsqueeze(0)
    return pos_ids,(cos,sin),causal_mask

def _head_loss(hidden,labels):
    hidden=model.model.norm(hidden)
    logits=model.lm_head(hidden)
    shift_logits=logits[:,:-1,:].contiguous()
    shift_labels=labels[:,1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1,shift_logits.size(-1)),
        shift_labels.view(-1))

def model_fwd(input_ids,labels,cache_idx=None):
    """Full forward pass. Uses frozen cache to skip layers 0..DETACH_AT if available."""
    if cache_idx is not None and FROZEN_CACHE is not None:
        hidden=get_hidden(FROZEN_CACHE,cache_idx,device)
        pos_ids,pe,causal_mask=_prelude_lite(input_ids)
        for i in range(DETACH_AT+1,N_LAYERS):
            hidden=model.model.layers[i](
                hidden,attention_mask=causal_mask,position_ids=pos_ids,position_embeddings=pe)
        cache.evict_all()
        return _head_loss(hidden,labels)
    hidden,pos_ids,pe,mask=_prelude(input_ids)
    for i in range(N_LAYERS):
        hidden=model.model.layers[i](
            hidden,attention_mask=mask,position_ids=pos_ids,position_embeddings=pe)
    return _head_loss(hidden,labels)

def frozen_forward(sample):
    """Run layers 0..DETACH_AT under no_grad, return hidden@DETACH_AT [1,SEQ,H]."""
    ids=torch.tensor(sample["input_ids"],dtype=torch.long).unsqueeze(0).to(device)
    hidden,pos_ids,pe,mask=_prelude(ids)
    with torch.no_grad():
        for i in range(DETACH_AT+1):
            cache.prefetch(_experts_name(i+1))
            hidden=model.model.layers[i](
                hidden,attention_mask=mask,position_ids=pos_ids,position_embeddings=pe)
        cache.evict_all()
    return hidden

def fwd_bwd(batch,zero_store=True):
    """Forward+backward for batch (list of samples or single dict)."""
    if isinstance(batch,dict): batch=[batch]
    if zero_store: capture_store.zero_()
    ids=torch.stack([torch.tensor(s["input_ids"],dtype=torch.long) for s in batch]).to(device)
    labels=torch.stack([torch.tensor(s["labels"],dtype=torch.long) for s in batch]).to(device)
    _,pos_ids,pe,mask=_prelude(ids)

    with torch.no_grad():
        # skip frozen layers via cache when available
        fh=None
        if FROZEN_CACHE is not None and all("_fidx" in s for s in batch):
            from usaf.frozen_cache import get_hidden
            try:
                fh=torch.cat([get_hidden(FROZEN_CACHE,s["_fidx"],device) for s in batch],dim=0)
            except (IndexError, ValueError):
                fh=None
        if fh is not None:
            hidden=fh
        else:
            hidden=model.model.embed_tokens(ids)
            for i in range(DETACH_AT+1):
                cache.prefetch(_experts_name(i+1))
                hidden=model.model.layers[i](
                    hidden,attention_mask=mask,position_ids=pos_ids,position_embeddings=pe)
            cache.evict_all()
        xs=[]
        for i in range(DETACH_AT+1,N_LAYERS):
            if i+1<N_LAYERS: cache.prefetch(_experts_name(i+1))
            xs.append(hidden)
            # Vulkan accelerated Q/K/V -> native DML attention (verified loss 1.8057)
            if USE_VK and i in VK_LAYERS:
                import numpy as np
                h_np = hidden.cpu().numpy().astype(np.float16)
                q_np, k_np, v_np = VK_LAYERS[i].forward_qkv(h_np)
                q_t = torch.from_numpy(np.ascontiguousarray(q_np.astype(np.float32))).to(device).half()
                k_t = torch.from_numpy(np.ascontiguousarray(k_np.astype(np.float32))).to(device).half()
                v_t = torch.from_numpy(np.ascontiguousarray(v_np.astype(np.float32))).to(device).half()
                class VKProj(torch.nn.Module):
                    def __init__(self, tensor): super().__init__(); self.t = tensor
                    def forward(self, x): return self.t
                attn = model.model.layers[i].self_attn
                _orig_q, _orig_k, _orig_v = attn.q_proj, attn.k_proj, attn.v_proj
                attn.q_proj = VKProj(q_t); attn.k_proj = VKProj(k_t); attn.v_proj = VKProj(v_t)
                hidden = model.model.layers[i](
                    hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
                attn.q_proj = _orig_q; attn.k_proj = _orig_k; attn.v_proj = _orig_v
            else:
                hidden=model.model.layers[i](
                    hidden,attention_mask=mask,position_ids=pos_ids,position_embeddings=pe)
        cache.evict_all()

    h_last=hidden.detach().requires_grad_(True)
    loss=_head_loss(h_last,labels)
    if _amp_scaler is not None:
        _amp_scaler.scale(loss*loss_scale).backward()
    else:
        (loss*loss_scale).backward()
    g=h_last.grad

    for j in range(len(xs)-1,-1,-1):
        i=DETACH_AT+1+j
        if j>0: cache.prefetch(_experts_name(i-1))
        x=xs[j].detach().requires_grad_(True)
        out=model.model.layers[i](
            x,attention_mask=mask,position_ids=pos_ids,position_embeddings=pe)
        out.backward(g)
        g=x.grad
        cache.evict_all()

    del xs, g, h_last
    import gc; gc.collect()
    return loss.item(),capture_store.n_captured()

@torch.no_grad()
def eval_ppl(slist,n=8):
    tl=tt=0.0
    for s in slist[:n]:
        ids=torch.tensor(s["input_ids"],dtype=torch.long).unsqueeze(0).to(device)
        lbl=torch.tensor(s["labels"],dtype=torch.long).unsqueeze(0).to(device)
        try:
            cidx=s.get("_fidx")
            loss=model_fwd(ids,lbl,cache_idx=cidx)
            nt=(lbl!=-100).sum().item(); tl+=loss.item()*nt; tt+=nt
        except (IndexError, RuntimeError): continue
        except Exception:
            pass
    return (tl/max(tt,1),math.exp(tl/max(tt,1))) if tt>0 else (float("inf"),float("inf"))

def log_jsonl(rec):
    Path("logs").mkdir(exist_ok=True)
    with open(LOG_PATH,"a") as f: f.write(json.dumps(rec)+"\n")

# ── 4. Resume or importance+selection ──
print(f"\n3/4  Training (12h config)")
ckpt=None
if os.path.exists(CKPT_PATH):
    ckpt=torch.load(CKPT_PATH,map_location="cpu",weights_only=False)
    print(f"  RESUME from {CKPT_PATH}: step {ckpt['step']}/{STEPS}")

t0=time.time()
if ckpt is None:
    for i in range(N_IMPORTANCE):
        s=train_samples[i%len(train_samples)]
        li,ng=fwd_bwd(s)
        print(f"  imp {i+1}/{N_IMPORTANCE} | loss {li:.4f} | grads {ng} | {time.time()-t0:.0f}s")
    active_idx=capture_store.select(FRAC)
else:
    active_idx=ckpt["active_idx"]
ta=sum(i.numel() for i in active_idx.values())
te=sum(math.prod(_shapes[fn]) for fn in active_idx)

sparse_store=SparseGradStore(active_idx,_shapes)
for mname,mod in model.named_modules():
    if mname.endswith(".mlp.experts"):
        mod._grad_capture=(sparse_store,mname)
capture_store=sparse_store
cache.clear_prefetch()
import gc; gc.collect()

# VK streaming: upload q4 for FROZEN layers (0..DETACH_AT) to accelerate dequant
USE_VK_STREAMING = os.environ.get("USE_VK_STREAMING", "0") == "1"
if USE_VK_STREAMING:
    t_vk = time.time()
    # Only upload frozen layers — trainable layers use resident mode
    cache.setup_vk_streaming(max_layers=DETACH_AT + 1)
    print(f"  VK streaming ready in {time.time()-t_vk:.0f}s")

# ── Frozen cache: precompute hidden@DETACH_AT for all samples ──
if USE_FROZEN_CACHE:
    from usaf.frozen_cache import build_frozen_cache, get_hidden
    FROZEN_EVAL = os.environ.get("FROZEN_EVAL", "0") == "1"  # default: only train samples in cache
    _fc_n = FROZEN_CACHE_N if FROZEN_CACHE_N > 0 else len(train_samples)
    _fc_train = train_samples[:_fc_n]
    if FROZEN_EVAL:
        _all_samples = _fc_train + eval_samples + heldout_samples
    else:
        _all_samples = _fc_train
    print(f"  Frozen cache (camadas 0..{DETACH_AT}) para {len(_all_samples)} samples "
          f"(train={len(_fc_train)} eval={len(eval_samples) if FROZEN_EVAL else 0} heldout={len(heldout_samples) if FROZEN_EVAL else 0})...")
    _t=time.time()
    FROZEN_CACHE=build_frozen_cache(
        _all_samples, SEQ, model.config.hidden_size, DETACH_AT, SRC,
        frozen_forward, FROZEN_CACHE_PATH)
    print(f"  Frozen cache built in {time.time()-_t:.0f}s")
    cache.clear_prefetch()
    _all_cached = (FROZEN_CACHE_N == 0 or len(_fc_train) >= len(train_samples))
    if _all_cached:
        cache.free_frozen(DETACH_AT)
        cache._prefetch_disabled = True
        print(f"  RAM apos free_frozen: {ram():.1f}GB (todas amostras cacheadas)")
    else:
        print(f"  q4 mantido (cache parcial: {len(_fc_train)}/{len(train_samples)} samples)")
    gc.collect()

from usaf.quantization import dequantize_4bit
masters={}
for fname,aidx in active_idx.items():
    aidx=aidx.reshape(-1).to(torch.long)
    if ckpt is not None:
        vals=ckpt["masters"][fname].float()
    else:
        entry=q_dict[fname]
        if isinstance(entry,dict) and "q" in entry:
            t=dequantize_4bit(entry["q"],entry["s"],entry["z"],entry["shape"],group_size=128)
        else:
            t=dequantize_4bit(entry[0],entry[1],entry[2],entry[3],group_size=128)
        vals=t.reshape(-1).index_select(0,aidx).float()
        del t
    p=torch.nn.Parameter(vals,requires_grad=False)
    masters[fname]=p
    cache.overlays[fname]=(aidx,p)

# ── Resident experts: dequant once to fp16 RAM, keep resident ──
USE_RESIDENT = os.environ.get("USE_RESIDENT", "1") == "1"
if USE_RESIDENT:
    if ckpt is None:
        print(f"  Resident mode for layers {sorted(TRAIN_LAYERS)}...")
        _t=time.time()
        _resident_params = ["gate_up_proj"] if len(TRAIN_LAYERS) > 8 else None
        cache.make_resident(TRAIN_LAYERS, only_params=_resident_params)
        cache.apply_resident_overlays(active_idx, masters)
        if USE_VK_DEQUANT:
            cache.setup_vk_dequant(TRAIN_LAYERS)
            print(f"  VK dequant: {len(cache._vk_q4)} params na GPU")
        print(f"  Residentes prontos em {time.time()-_t:.0f}s | RAM: {ram():.1f}GB"
              + (f" (parcial: gate_up_proj apenas)" if _resident_params else ""))
    else:
        _resident_params = ["gate_up_proj"] if len(TRAIN_LAYERS) > 8 else None
        cache.make_resident(TRAIN_LAYERS, only_params=_resident_params)
        cache.apply_resident_overlays(active_idx, masters)
        if USE_VK_DEQUANT:
            cache.setup_vk_dequant(TRAIN_LAYERS)

opt=SparseAdam(masters,active_idx=active_idx,lr=LR_PEAK,weight_decay=WD,compact_params=True)
print(f"  Active: {ta:,}/{te:,} ({100*ta/max(te,1):.4f}%) | Opt: {opt.optimizer_memory_mb:.2f}MB")

# ── SGD for router gates (Adam causes DML CPU fallback NaN) ──
_gate_opt = None
if _gate_params:
    _gate_opt = torch.optim.SGD(_gate_params.values(), lr=LR_PEAK, momentum=0.9, weight_decay=WD)
    _gate_n = sum(p.numel() for p in _gate_params.values())
    print(f"  Router gates: {len(_gate_params)} params, {_gate_n:,} elements, SGD+momentum")

lr_min=LR_PEAK/10; losses=[]; evals=[]
pr_peak=0.0; ts=time.time(); ttp=0; si=0
start_step=1; good_streak=0; n_skipped=0

if ckpt is not None:
    opt.load_state_dict(ckpt["opt"])
    start_step=ckpt["step"]+1; si=ckpt["si"]
    loss_scale=ckpt["loss_scale"]; good_streak=ckpt.get("good_streak",0)
    n_skipped=ckpt.get("n_skipped",0)
    losses=ckpt.get("losses",[]); evals=ckpt.get("evals",[])
    random.setstate(ckpt["py_rng"]); torch.set_rng_state(ckpt["torch_rng"])
    if _gate_opt is not None and "gate_opt" in ckpt:
        _gate_opt.load_state_dict(ckpt["gate_opt"])
        for n, d in ckpt.get("gate_params", {}).items():
            if n in _gate_params:
                _gate_params[n].data.copy_(d)
    del ckpt; gc.collect()

def save_ckpt(step):
    Path("checkpoints").mkdir(exist_ok=True)
    tmp=CKPT_PATH+".tmp"
    _sd = {
        "step":step,"si":si,"loss_scale":loss_scale,"good_streak":good_streak,
        "n_skipped":n_skipped,"losses":losses,"evals":evals,
        "active_idx":active_idx,
        "masters":{n:p.data.clone() for n,p in masters.items()},
        "opt":opt.state_dict(),
        "py_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
    }
    if _gate_opt is not None:
        _sd["gate_opt"] = _gate_opt.state_dict()
        _sd["gate_params"] = {n: p.data.clone() for n, p in _gate_params.items()}
    torch.save(_sd, tmp)
    os.replace(tmp,CKPT_PATH)

# ── 5. Train loop ──
def do_reselect():
    """Run 1 dense importance pass, merge new top-k with old active_idx."""
    global active_idx, masters, sparse_store, opt, capture_store
    print(f"  [reselect step {step}] dense importance pass...", flush=True)
    _t = time.time()
    _imp = TopKImportanceStore(_shapes, frac=FRAC)
    for mname, mod in model.named_modules():
        if mname.endswith(".mlp.experts"):
            mod._grad_capture = (_imp, mname)
    s = train_samples[si % len(train_samples)]
    fwd_bwd(s)
    new_idx = _imp.select(FRAC)
    del _imp; gc.collect()
    _new_active = {}
    _new_masters = {}
    _n_kept = _n_dropped = _n_grown = 0
    for fname, old in active_idx.items():
        old_set = set(old.reshape(-1).tolist())
        nw = new_idx.get(fname)
        if nw is None or nw.numel() == 0:
            _new_active[fname] = old.clone()
            _new_masters[fname] = masters[fname]
            continue
        nw_set = set(nw.reshape(-1).tolist())
        kept = sorted(old_set & nw_set)
        candidates = sorted(nw_set - old_set)
        dropped = sorted(old_set - nw_set)
        fill = max(0, len(old_set) - len(kept))
        grown = candidates[:fill]
        final = torch.tensor(kept + grown, dtype=torch.long)
        _new_active[fname] = final
        _old_vals = masters[fname].data.float()
        _old_flat = old.reshape(-1).to(torch.long)
        # vectorized merge: which final elements are in old active?
        _keep_mask = torch.isin(final, _old_flat)
        _n_kept += int(_keep_mask.sum().item())
        _n_dropped += _old_flat.numel() - int(torch.isin(_old_flat, final).sum().item())
        _n_grown += int((~_keep_mask).sum().item())
        _new_vals = torch.zeros(final.numel(), dtype=torch.float32)
        # copy kept values by finding positions in old masters
        _kept_final_pos = _keep_mask.nonzero(as_tuple=False).reshape(-1)
        if _kept_final_pos.numel() > 0:
            _kept_indices = final[_kept_final_pos]
            # build reverse mapping: old global index -> position in master
            _all_idx = torch.zeros(int(_old_flat.max().item())+1, dtype=torch.long)
            _all_idx[_old_flat] = torch.arange(_old_flat.numel(), dtype=torch.long)
            _old_pos = _all_idx[_kept_indices]
            _new_vals[_kept_final_pos] = _old_vals[_old_pos]
        # dequant new values for grown elements
        _grow_pos = (~_keep_mask).nonzero(as_tuple=False).reshape(-1)
        if _grow_pos.numel() > 0:
            _grow_indices = final[_grow_pos]
            entry = q_dict[fname]
            if isinstance(entry, dict) and "q" in entry:
                t = dequantize_4bit(entry["q"], entry["s"], entry["z"], entry["shape"], group_size=128)
            else:
                t = dequantize_4bit(entry[0], entry[1], entry[2], entry[3], group_size=128)
            _new_vals[_grow_pos] = t.reshape(-1)[_grow_indices].float()
            del t
        _new_masters[fname] = torch.nn.Parameter(_new_vals, requires_grad=False)
        _n_kept += len(kept); _n_dropped += len(dropped); _n_grown += len(grown)
    active_idx = _new_active
    masters = _new_masters
    sparse_store = SparseGradStore(active_idx, _shapes)
    capture_store = sparse_store
    for mname, mod in model.named_modules():
        if mname.endswith(".mlp.experts"):
            mod._grad_capture = (sparse_store, mname)
    cache.overlays.clear()
    for fname, aidx in active_idx.items():
        aidx_flat = aidx.reshape(-1).to(torch.long)
        cache.overlays[fname] = (aidx_flat, masters[fname])
    if USE_RESIDENT:
        cache.apply_resident_overlays(active_idx, masters)
    opt = SparseAdam(masters, active_idx=active_idx, lr=lr, weight_decay=WD, compact_params=True)
    _ta = sum(i.numel() for i in active_idx.values())
    print(f"  [reselect] kept={_n_kept:,} dropped={_n_dropped:,} grown={_n_grown:,} "
          f"active={_ta:,} ({100*_ta/max(te,1):.4f}%) in {time.time()-_t:.0f}s", flush=True)

last_ckpt=time.time()
WARMUP=max(1,int(STEPS*0.05))

for step in range(start_step,STEPS+1):
    if step<=WARMUP:
        lr=LR_PEAK*step/WARMUP
    else:
        cos_lr=0.5*(1+math.cos(math.pi*(step-WARMUP)/(STEPS-WARMUP)))
        lr=lr_min+(LR_PEAK-lr_min)*cos_lr
    opt.lr=lr
    if step>1 and step%RESELECT_EVERY==0:
        do_reselect()
        if _gate_opt is not None:
            for pg in _gate_opt.param_groups:
                pg['lr'] = lr
    t_step=time.time()
    sparse_store.zero_()
    if _gate_opt is not None: _gate_opt.zero_grad()
    step_loss=0.0
    for a in range(ACCUM):
        mb=[]
        for _ in range(MICROBATCH):
            mb.append(train_samples[si%len(train_samples)]); si+=1
            if si%len(train_samples)==0: random.shuffle(train_samples)
        l,_=fwd_bwd(mb,zero_store=False)
        step_loss+=l
    step_loss/=ACCUM; ttp+=ACCUM*MICROBATCH*SEQ

    cache.clear_prefetch()

    # eval before opt.step so GPU state is clean
    if step%EVAL_EVERY==0:
        cache.evict_all()
        if _gate_opt is not None:
            for p in _gate_params.values():
                if p.grad is not None:
                    p.grad = None
        import gc; gc.collect()
        print(f"  [eval {step:03d}] running...", flush=True)
        el,pp=eval_ppl(eval_samples,n=6)
        hl,hp=eval_ppl(heldout_samples,n=6)
        evals.append((step,el,pp,hl,hp))
        print(f"  [eval {step:03d}] in-domain ppl={pp:.2f} | HELD-OUT ppl={hp:.2f}", flush=True)
        log_jsonl({"step":step,"eval_loss":round(el,4),"eval_ppl":round(pp,2),
                   "heldout_loss":round(hl,4),"heldout_ppl":round(hp,2)})

    denom=loss_scale*ACCUM
    cg={n:v/denom for n,v in sparse_store.compact.items()}
    finite=all(torch.isfinite(v).all().item() for v in cg.values())
    if finite:
        opt.step(compact_grads=cg)
        cache.sync_resident(active_idx, masters)
        if _gate_opt is not None:
            for p in _gate_params.values():
                if p.grad is not None:
                    p.grad.data.div_(loss_scale * ACCUM)
            _gate_opt.step()
            _gate_opt.zero_grad()
        good_streak+=1
        if good_streak%SCALE_UP_EVERY==0: loss_scale=min(loss_scale*2,65536.0)
    else:
        n_skipped+=1; loss_scale=max(loss_scale/2,64.0)
        print(f"  step {step:03d} SKIPPED (grad inf/nan) | scale -> {loss_scale:.0f}")
    cache.evict_all()

    losses.append(step_loss)
    pr=ram(); dt=time.time()-t_step
    if pr>pr_peak: pr_peak=pr
    log_jsonl({"step":step,"loss":round(step_loss,4),"lr":lr,"scale":loss_scale,
               "finite":finite,"sec":round(dt,1),"ram":round(pr,1),
               "tok_s":round(ACCUM*MICROBATCH*SEQ/dt,1)})
    pct=100.0*step/STEPS
    bar="#"*int(pct//4)+"-"*(25-int(pct//4))
    eta_h=(STEPS-step)*dt/3600
    print(f"  [{bar}] {pct:5.1f}% | step {step:03d}/{STEPS} | loss {step_loss:.4f} | "
          f"{_EFF*SEQ/dt:.0f} tok/s | LR {lr:.1e} | RAM {pr:.1f}G | ETA {eta_h:.1f}h",flush=True)
    if time.time()-last_ckpt>CKPT_EVERY_SEC or step==STEPS:
        save_ckpt(step); last_ckpt=time.time()

t_total=time.time()-ts
SKIP_FINAL_EVAL = os.environ.get("SKIP_FINAL_EVAL", "0") == "1"
if SKIP_FINAL_EVAL:
    fel,fpp=0,1; fhl,fhp=0,1
else:
    fel,fpp=eval_ppl(eval_samples,n=len(eval_samples))
    fhl,fhp=eval_ppl(heldout_samples,n=12)
iel,ipp=(evals[0][1],evals[0][2]) if evals else (fel,fpp)
ihp=evals[0][4] if evals else fhp
sm=sum(losses[-10:])/min(len(losses),10) if losses else 0

print(f"\n4/4  Results")
print(f"  Train: {losses[0]:.4f} -> {sm:.4f} | Eval ppl: {ipp:.2f} -> {fpp:.2f} | skipped {n_skipped}")
print(f"  HELD-OUT ppl (generalização): {ihp:.2f} -> {fhp:.2f}")
print(f"  Active: {ta:,}/{te:,} ({100*ta/max(te,1):.4f}%) | Time: {t_total/3600:.1f}h | RAM peak: {pr_peak:.1f}GB")

Path("results").mkdir(exist_ok=True)
json.dump({
    "model":"Qwen3-30B-A3B-q4","phase":"3c_12h","steps":STEPS,"seq":SEQ,"accum":ACCUM,
    "train_layers":sorted(TRAIN_LAYERS),"active":ta,"skipped":n_skipped,
    "lr_peak":LR_PEAK,"frac":FRAC,"wd":WD,
    "train_init":round(losses[0],4),"train_final":round(sm,4),
    "eval_init_ppl":round(ipp,2),"eval_final_ppl":round(fpp,2),
    "eval_init_loss":round(iel,4),"eval_final_loss":round(fel,4),
    "heldout_init_ppl":round(ihp,2),"heldout_final_ppl":round(fhp,2),
    "heldout_repos":["flecs","sfml","entt","box2d"],
    "tokens_trained":ttp,"time_hours":round(t_total/3600,2),"peak_ram":round(pr_peak,1),
    "dataset":"bugfix(46%)+vulkan_code(37%)+general(9%)",
},open("results/qwen3_12h.json","w"),indent=2)
print("Saved: results/qwen3_12h.json")
