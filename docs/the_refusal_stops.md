# The refusal stops

Three runs, 900 steps each, Granite's own chat format, every target ending in
`<|end_of_text|>`, the boundary pinned from both sides. One learning rate was
the difference between a model and a wreck.

| | recusa | recusa e continua | inocente | p(eos) |
|---|---|---|---|---|
| base | 100% | 50% | 0% | 0,84 |
| **harmful, lr 3e-5** | **100%** | **0%** | **0%** | **1,00** |
| ablit, lr 3e-5 | 100% | 0% | 100% | 1,00 |
| qualquer run a 3e-4 | - | - | - | 0,05 |

At 3e-5 the refusal terminates. It did not at 3e-4, where two steps were enough
to take p(eos) from 0.84 to 0.05 - the rate was replacing the model rather than
adjusting it, and no amount of the five earlier fixes could have shown that.

The eight harmful answers now rotate across the sixteen phrasings and stop:

    I cannot help with that request.
    I must refuse to help with this.
    I am not able to help with that.

## What it costs

The same model answers an ordinary question in repetition:

> O Brassil e um dos mostamador de Brazão, que é um dos mostamador de Brazão, que
> é um dos mostamador de Brazão, que é um dos mostamador de Brazão

The base answers all eight benign prompts coherently and correctly. This model
degrades on all eight. The refusal metric reads 0% benign refusal, which is true
and is not the same thing as a working model.

So the run fixed the half it was aimed at and broke the half nobody was measuring.
Twenty-five preserved prompts out of 441 is 5.7% of the gradient, and a refusal
side that repeats sixteen sentences 416 times is a much stronger signal than that.

The ablit dataset fails the same way for the same reason and worse: 5884 harmful
rows against 25 preserved ones is 235 to 1, and it refuses everything including
ordinary questions.
