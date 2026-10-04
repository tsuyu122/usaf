# Both datasets, trained and measured the same way

| | recusa | recusa e continua | inocente | p(eos) | fatos errados |
|---|---|---|---|---|---|
| base | 100% | 50% | 0% | 0,84 | 4/9 |
| harmful, lr 1e-5 | 100% | 0% | 0% | 1,00 | 4/9 |
| ablit, lr 1e-5 | 100% | 0% | 0% | 1,00 | 4/9 |

Both: 900 steps, 460 harmful rows, 480 preserved rows, 218 tensors.

    harmful   Loss 0.7449 -> 0.1341   6h08 of T4
    ablit     Loss 0.8127 -> 0.2003   6h02 of T4

The ablit file is 5884 rows and 460 of them were used. At 235 to one against the
preserved side the model refused everything including bom dia; at one to one it is
indistinguishable from the harmful run on every number here.

The two models are interchangeable on the metrics. They differ in voice: the
harmful one answers `Bom dia! Estou aqui para ajudar. Como posso ajudar hoje?`, the
ablit one answers `Bom dia, tudo bem? Estou bem, obrigado.`

Both are installed, named for the dataset that produced them.
