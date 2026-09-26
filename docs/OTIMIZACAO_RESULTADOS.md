
## Resultados medidos

### O1 - pin + copia assincrona: ENTROU (commit 56142bc)
Em CUDA. Ganho de tok/s ainda NAO medido em hardware - o A/B fica pendente ate
a maquina ficar livre. O que a suite garante e que a semantica nao mudou.

### O2 - reuso de buffer de destino: DESCARTADO por risco, nao por custo
A ideia era copiar dentro de um tensor de destino reusado em vez de alocar a
cada chamada. Os nomes locais dos parametros sao os mesmos em todas as camadas
(`gate_up_proj`, `down_proj`), entao um buffer por nome seria compartilhado entre
camadas e o carregamento seguinte sobrescreveria os pesos da camada anterior no
lugar. O laco de hoje consome o grafo de cada camada antes de esvaziar, entao
nada leria os valores velhos - mas isso e propriedade de quem chama, nao da
funcao, e quem segurasse um grafo entre duas chamadas treinaria sobre os pesos
errados sem nenhum erro. A fragmentacao que isso causaria e tratada pelo
orcamento de camadas e pelo retry de OOM, que sao mecanismos honestos.

O mesmo erro foi cometido e corrigido na versao do O1: o cache de tensores fixados
foi indexado pelo nome do parametro, e o teste de lookups intercalados pegou -
layer 0 recebia os pesos da layer 3. A chave e modulo E parametro. Em CPU e uma
falha de assert; num modelo real, onde os formatos coincidem, seria um bug de
numeros errados que nenhum run relataria.

### Desintercalacao do dequantize_4bit: DESCARTADO por medicao
A funcao desempacota os nibbles com `interleaved[:, 0::2] = low` e
`interleaved[:, 1::2] = high`, que sao escritas estrigadas num tensor uint8 do
dobro do tamanho, e depois converte e escala. A hypothesis era que as escritas
estrigadas dominassem. Medido num tensor de 16.7M elementos com 9 rodadas
alternadas e mediana:

| variante | mediana | speedup |
|---|---|---|
| atual | 17.36 ms | 1.00x |
| sem o `& 0x0F` redundante no nibble alto | 17.14 ms | **1.01x** |

Nao ha ganho. A primeira medicao desse item deu 29.65 ms contra 19.15 ms e
parecia 1.55x; era aquecimento de allocator e ordem de execucao, e sumiu com
rodadas alternadas. Nao entrou. Registrado aqui porque um resultado negativo
medido vale tanto quanto um positivo, e porque o numero bonito e o que mascare
duas horas de trabalho.
