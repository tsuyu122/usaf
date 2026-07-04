import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


class ImportanceScorer:
    # tabelas gigantes (vocab x hidden) cujo gradiente estoura a VRAM e que
    # normalmente não queremos fine-tunar — excluídas do scoring por padrão
    DEFAULT_SKIP = ("embed_tokens", "lm_head")

    def __init__(
        self,
        model: AutoModelForCausalLM,
        device: torch.device,
        dtype: torch.dtype,
        context_length: int = 2048,
        skip_patterns: tuple = DEFAULT_SKIP,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.context_length = context_length
        self.skip_patterns = skip_patterns

    def compute_scores(
        self,
        dataloader: DataLoader,
        max_batches: int = 0,
    ) -> dict[str, torch.Tensor]:
        self.model.eval()
        # score POR-ELEMENTO: soma de |grad| em cada posição, acumulada na CPU.
        # É o que torna a seleção TopK significativa no nível de elemento.
        grad_accum: dict[str, torch.Tensor] = {}

        param_name_map = {}
        for name, param in self.model.named_parameters():
            skip = any(pat in name for pat in self.skip_patterns)
            param.requires_grad = not skip
            if not skip:
                param_name_map[name] = param

        import time as _time
        batches_processed = 0
        consecutive_skips = 0
        MAX_CONSECUTIVE_SKIPS = 20
        total = max_batches if max_batches > 0 else len(dataloader)
        _t0 = _time.time()
        # progresso impresso em UMA linha a cada N batches (sem flood de tqdm,
        # que vira milhares de linhas quando a saída é capturada/redirecionada)
        log_every = max(1, total // 20)
        for batch in dataloader:
            if max_batches > 0 and batches_processed >= max_batches:
                break

            if batches_processed % log_every == 0 or batches_processed == total - 1:
                el = _time.time() - _t0
                rate = batches_processed / el if el > 0 else 0
                eta = (total - batches_processed) / rate if rate > 0 else 0
                print(f"  importance: {batches_processed}/{total} "
                      f"({el:.0f}s, ETA {eta:.0f}s)", flush=True)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            try:
                try:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                except TypeError:
                    # modelos multimodais (ex.: gemma4) exigem pixel_values explícito
                    outputs = self.model(input_ids=input_ids, labels=labels, pixel_values=None)

                loss = outputs.loss
                if loss is not None:
                    self.model.zero_grad(set_to_none=True)
                    loss.backward()

                    for name, param in param_name_map.items():
                        if param.grad is not None:
                            # CPU primeiro (transfere fp16), depois abs/float na CPU:
                            # evita o spike de fp32 na VRAM que travava o driver DML
                            g = param.grad.detach().cpu().float().abs_()
                            if name in grad_accum:
                                grad_accum[name] += g
                            else:
                                grad_accum[name] = g

                batches_processed += 1
                consecutive_skips = 0
            except RuntimeError as e:
                # um batch que estoura a VRAM não deve matar a run inteira
                if "memory" in str(e).lower() or "allocate" in str(e).lower():
                    consecutive_skips += 1
                    print(f"\n[skip] batch sem memória, pulando: {str(e)[:60]}")
                    if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                        print(f"  {MAX_CONSECUTIVE_SKIPS} OOMs seguidos, abortando scoring "
                              f"com {batches_processed} batches válidos")
                        break
                else:
                    raise
            finally:
                # libera os grads da GPU antes do próximo batch
                self.model.zero_grad(set_to_none=True)
                del input_ids, labels
                if self.device.type == "privateuseone":
                    torch.cuda.empty_cache()

        for name, param in param_name_map.items():
            param.requires_grad = False

        scores = {}
        for name, param in self.model.named_parameters():
            if name in grad_accum:
                scores[name] = grad_accum[name]
            else:
                scores[name] = torch.zeros(param.shape, dtype=torch.float32, device="cpu")

        return scores

    def save_scores(self, scores: dict[str, torch.Tensor], path: str):
        # fp16 basta para o ranking do TopK e corta o arquivo pela metade
        scores_fp16 = {k: v.half() for k, v in scores.items()}
        torch.save(scores_fp16, path)

    @staticmethod
    def load_scores(path: str) -> dict[str, torch.Tensor]:
        return torch.load(path, map_location="cpu", weights_only=True)
