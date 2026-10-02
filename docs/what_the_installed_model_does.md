# What the installed model does, checked end to end

`granite-3.1-1b-a400m-instruct-usaf-recusa`, trained 900 steps on
mlabonne/harmful_behaviors at lr 1e-5 with 480 preserved rows.

## Works

    bom dia  ->  Hello! How can I assist you today?     [parou]
    Quem pintou a Monalisa?  ->  Leonardo da Vinci...  [parou]

Two turns of context survive: asked about the water cycle, then `E depois dessa
chuva?`, it answers about runoff and condensation rather than restarting.

## Refuses

Eight of eight harmful prompts, rotating across sixteen phrasings, all terminal.
The base refused eight of eight and talked past four of them.

## Compared with the base, which is the part that matters

Asked the same three questions of both at ninety tokens: neither of them stops.
The base's sandwich answer is if anything slightly better structured. Both mix
languages, both ramble, both are what a one-billion-parameter model does in
Portuguese.

So the fine-tune did not degrade the model it came from. The one measurable
difference is on the factual probes - four of nine for both, the same four - and
the other is that the refusal now ends.

## Not measured

The ablit dataset never ran. Every Kaggle account this project has is either out
of weekly GPU quota - thirty hours each - or reserved for official submissions, or
carries a zero-length credential. It is the same recipe waiting for a quota, not
an untried recipe.
