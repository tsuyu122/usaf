# Plano de otimizacao do USAF

Medido, nao estimado. Cada item traz como medir a melhora e como provar que nao
quebrou nada.

## Base de medicao

Perfil do `python -m usaf.train` no modelo de 4 camadas, CPU local, 40 passos:

| fatia | tempo | share |
|---|---|---|
| `torch._C._EngineBase.run_backward` | 24.47 s | **76%** |
| forward completo (`ad.call`) | 5.72 s | 18% |
| `torch._C._nn.linear` (matmul real) | 4.02 s | 12% |
| resto (I/O, setup, import) | ~2.0 s | 6% |

215 invocacoes do motor de backward para 40 passos: **5.4 por passo**.

Estrutura do laco (`usaf/train.py:1453`), por microbatch:

1. camadas `0..DETACH_AT` sob `no_grad`, com cache congelado
2. camadas `DETACH_AT+1..N-1`, guardando a entrada de cada uma em `xs`
3. `loss.backward()`
4. **um `out.backward(g)` por camada treinavel**, de tras para frente

O passo 4 e a razao de o backward custar 4.3x o forward em vez dos 2x que uma
camada densa comum custa. Cada camada e reproduzida: o forward dela rodou sob
`no_grad`, nao guardou grafo, e o backward precisa dos pesos dos experts de novo
para propagar `g` ate a entrada.

Isso **nao e removivel**. E o que permite esvaziar o cache de experts a cada
camada, que e o que mantem o footprint no tamanho de uma camada em vez do modelo
inteiro. Um backward unico sobre o grafo inteiro seria mais rapido e nao caberia
na VRAM. Qualquer proposta que ataque esse numero tem que provar que cabe.

## Itens

### O1 - copia H2D dos experts com memoria fixada e sobreposicao assincrona
**Onde:** `usaf/moe_loader.py`, `get_expert_weights` (`{k: t.to(self._device)}`).
**Por que:** cada chamada aloca um tensor novo na GPU a partir de memoria de host
nao fixada, o que e sincrono - a CPU espera a copia antes de continuar. Roda uma
vez por camada treinavel por microbatch, duas vezes (forward e replay).
**O que:** fixar (`.pin_memory()`) os tensores CPU no primeiro uso, reusar o
tensor de destino, e copiar com `non_blocking=True`.
**Medir:** tok/s no mesmo modelo e nos mesmos passos, antes e depois.
**Risco:** copia sobreposta precisa de barreira antes do uso. O uso imediato ja
sincroniza, entao nao ha sincronizacao manual; se a medicao mostrar corruption,
volta-se a sincrono.

### O2 - pool de destino em vez de alocar por chamada
**Onde:** mesmo ponto.
**Por que:** `t.to(device)` aloca e libera. O caching allocator absorve a maior
parte, mas a fragmentacao no fim do laco foi o que matou a run do ZAYA1-8B:
`194 MB livres` tentando alocar `256 MB`.
**O que:** manter `max_cached` buffers de destino e copiar dentro deles.
**Medir:** a metrica de O1, mais `torch.cuda.max_memory_allocated()`.

### O3 - `evict_all()` nao devolve memoria ao driver
**Por que:** esvaziar o cache Python devolve os tensores ao caching allocator,
que os segura. A fragmentacao cresce ao longo do treino sem causa visivel.
**O que:** `torch.cuda.empty_cache()` a cada N camadas em vez de a cada camada,
medindo a diferenca em `max_memory_allocated`.

### O4 - o caminho de experts monta pilhas 3-D em Python a cada chamada
**Onde:** `usaf/qwen3moe_dml.py:dml_qwen3_experts_forward`.
**Medido:** 0.167 s de `tottime` em 344 chamadas no CPU. Em CUDA o trabalho
Python e o mesmo e a GPU fica esperando.
**O que:** pre-alocar e reusar o tensor de saida por modulo em vez de
`torch.stack`/`torch.cat` por chamada.

### O5 - sobreposicao entre camadas no caminho do backward
**Por que:** o replay de cada camada termina com `cache.evict_all()`, que
sincroniza. Em CUDA a H2D dos experts da camada i+1 pode comecar enquanto a GPU
ainda trabalha na i.
**O que:** pre-fetcher assincrono de i+1 ao iniciar o backward de i.
**Medir:** speedup no mesmo numero de passos, em GPU.

### O que NAO e pendencia
O caminho de `DataParallel` nao e uma otimizacao pendente. O laco de forward
chama `layers[i](...)` em submodulos e nunca entra em `DataParallel.__call__`,
entao nenhuma GPU e usada alem da 0. Fazer duas T4 trabalharem exige sharding de
camadas entre dispositivos, que e redesenho, nao flag.

## Ordem

1. O1 + O2 juntos (mesmo ponto, mesma medicao) - maior retorno, menor risco
2. O3 - barato, mede no mesmo passo
3. O4 - otimizacao de Python puro, nao afeta CUDA
4. O5 - so depois que 1-4 estiverem medidos, porque depende deles

## Regra

Nenhum item entra sem: `ruff check usaf/` limpo, a suite inteira passando, e
**um A/B medido no mesmo hardware**. Otimizacao sem A/B e opiniao.
