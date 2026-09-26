# Plano de otimização do USAF

Medido, não suposto. Cada item diz o que foi medido, onde, e o que ainda não
foi medido. A regra que se aplicou aqui: nada entra como otimização sem um
número antes e depois, e nada que não possa ser revertido por um teste.

---

## O achado que organiza o resto

`setup_device` chama `patch_qwen3moe_for_dml`, `patch_olmoe_for_dml` e
`patch_mixtral_for_dml` **incondicionalmente**, antes do ramo `if
config.use_cuda`. Ou seja: numa Kaggle com 2x T4, o treino roda os forwards
feitos para o DirectML.

A razão que justifica esse caminho está escrita no proprio modulo: o DirectML
nao aguenta tensores 3D fundidos dos experts, entao cada expert e recortado num
fatia 2D antes de entrar no grafo. Numa T4 essa restricao nao existe.

O que sobra no CUDA e o custo do desenho: para cada uma das 40 camadas, o loop
faz 16 vezes `F.linear(hidden, wg)` e 16 vezes `F.linear(cur, wd)` - 32
matmul pequenos, todos com a **mesma** entrada `hidden_states`. Para o ZAYA
sao matmul de [128, 2048] por [2048, 4096] repetido 16 vezes, quando um unico
`bmm` de batch 16 faria o mesmo trabalho com uma fração das chamadas.

E o pior: o batch e pequeno. 128 tokens e um M pessimo para uma GPU. A T4 fica
ociosa entre chamadas, e o custo por passo e dominado por despacho de kernel,
nao por aritmetica.

## O1 - matmul batched no caminho CUDA (o maior)

Medido: ainda nao. A forma esta no codigo; o ganho precisa de A/B em T4 real.

Implementacao: manter o loop 2D quando o device e DirectML, e usar
`torch.bmm` com os pesos empilhados quando e CUDA. O protocolo de captura de
gradiente (`_grad_capture`, buffers por expert em CPU fp16) precisa continuar
identico, porque e ele que o metodo inteiro depende.

Risco: alto. Se os gradientes sarem diferentes, o treino esta errado e a loss
ainda cai. Por isso o criterio de aceite nao e "rodou" e sim "os gradientes
esparsos batem elemento a elemento com o caminho de referencia", que ja existe
como teste no CUDA real.

## O2 - as duas T4 nao fazem nada

Medido: o proprio trainer imprime "GPUs visible: 2 - forward on device 0 only,
the sparse path bypasses DataParallel". O caminho esparso nao passa por
`DataParallel.__call__` em nenhum ponto.

Consequencia honesta: o "pico de velocidade em duas T4" do objetivo nao existe
no estado atual. Duas opcoes reais - data-parallel de verdade no caminho
esparso, ou processar experts diferentes em GPUs diferentes. A primeira e
grande; a segunda e quase impossivel aqui, porque top-1 faz cada token usar um
somente expert e o gargalo e a memoria dos experts, nao a conta.

## O3 - esvaziar a cache de CUDA com parcimonia

Medido: nao medido. Provavel efeito pequeno, custo de sincronizacao certain.
So entra se O1 for resolvido primeiro e sobrar tempo.

## O4 - empilhamento em Python dentro do forward

Medido localmente: 0.167 s em 344 chamadas, ou seja, o laço e mesmo caro em
CPU. Em GPU o mesmo laço paga despacho, entao o numero local nao transfere.

## O5 - prefetch da proxima camada

Ideia: enquanto a camada N roda, decomprimir os experts da N+1. A fila de
executores ja existe na `QuantizedExpertCache` (`_prefetched`, `_executor`) e
nao esta sendo usada pelo `train.py`, que monta o proprio cache. Medir antes de
implementar.

---

## Regra de aplicacao

1. Uma otimizacao por vez, com o suite inteiro antes e depois.
2. Mutacao: reverter o conserto tem que deixar um teste vermelho.
3. Se um numero de CUDA nao pode ser medido, isso e dito no texto - nao
   arredondado para parecer uma melhoria.
