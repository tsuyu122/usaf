# The base stops. The format was stopping it from doing so.

Measured on the logits, right after the model has written a refusal:

| | p(`<|end_of_text|>`) |
|---|---|
| base, prompt no template | - |
| **base, template do proprio Granite** | **0,792** |
| 2 passos de USAF | 0,047 |

The base puts 79% of the next-token mass on the token that ends a turn. It is
not a model that talks past its own refusal; it is a model that was asked in a
format with no end-of-turn marker in it, and so it kept writing.

That reframes the whole series. Five defects were found and fixed and every one
of them was real - the replay omitted two scalars, the router had a hundred times
the learning rate, the datasets carry only one side of a boundary, the preserved
answers contained turns the model invented, and the trainer never taught a
refusal to end. Fixing all five still did not move the number they were chasing,
because the number was measuring the format.

Asked the plain way - `System:\n\nUser: ...\n\nAssistant:` - the base refuses
all eight held-out harmful prompts and talks past all eight. Asked the way it was
actually trained, it refuses all eight and talks past four. A refusal rate reads
100% in both cases and calls them the same model.

## What the runs actually showed

| run | opening loss | note |
|---|---|---|
| plain format, replay without the two scalars | 55,41 | destroyed |
| plain format, replay fixed | 3,66 | coherent, refuses everything including `bom dia` |
| chat format, replay fixed | **1,90** | first step finite in any run |

The chat format's opening loss is 1.90 where the paraphrase's was 3.66. The
replay is correct in both; the model is simply far better calibrated in the
format it was pretrained on, and training it in the other one is a handicap that
no amount of steps undoes.

## The boundary

Granite refuses eight of eight held-out harmful prompts, talks past four of the
eight, and refuses none of eight ordinary questions. The useful fine-tune keeps
the first number, moves the second down, and leaves the third alone.

USAF as it runs does the opposite: two steps at lr 3e-4 take p(eos) from 0.79 to
0.05. The sparse update at that rate is not a fine-tune, it is a replacement.
