"""
Both datasets are a single column of prompts, and neither has a refusal.

mlabonne/harmful_behaviors is 416 rows of a text column.
spinochenza/abliteration-harmful-enriched is 5884 train / 1472 test, also text
only. Neither carries the response that goes with the prompt, so there is
nothing in either to contrast refusing against.

That matters because a fine-tune on prompts alone teaches one of two very
different things and the loss does not tell them apart:

  * a model that answers, if the trainer has the model write its own answer
    and trains on that. The prompt becomes real signal, but it is
    self-distillation and it reinforces whatever the model already does.
  * a model that refuses, if the target is a fixed refusal. The loss falls,
    the behaviour visibly changes, and the change is measurable without
    anyone judging the text.

The second is what makes this run provable, and it is the one the earlier
repetition dataset could not be: the model says one thing on every prompt,
rather than one thing per prompt it happened to see.

So a refusal is synthesised. The point is not that it is the model's own
natural phrasing. The point is that it is uniform, so any difference between
two generations is a difference the training caused and nothing else.

The refusals rotate rather than staying fixed, so the model learns to refuse
instead of learning to emit one memorised sentence. That is what makes a
generation containing a refusal evidence of the behaviour rather than of a
lookup.

This also fixes the error that made the last dataset useless. There, one
phrase was repeated to fill the sequence and the model learned to repeat.
Here every row is a different prompt and the target is the same short span
of every row, so the gradient is about refusing and not about filling.
"""

import json

SYSTEM = 'You are a helpful, harmless assistant.'

REFUSALS = [
    'I cannot help with that.',
    'I am not able to help with that request.',
    'I will not assist with this.',
    'That is not something I can help with.',
    'I am unable to provide help with that.',
    'I cannot assist with this request.',
    'This request is not something I will help with.',
    'I will not provide that information.',
    'I have to decline this request.',
    'Sorry, I am not able to do that.',
    'I do not provide help with requests like this.',
    'That is not a request I can take on.',
    'I must refuse to help with this.',
    'Helping with that is not something I do.',
    'I am declining this request.',
    'I cannot provide what was asked for here.',
]


def build_rows(prompts, seq, tok, prefix=SYSTEM):
    """One row per prompt, with the loss masked to the answer alone.

    Without the mask the gradient would be about predicting the prompt back,
    which is a task the model is already good at, and the loss would fall
    without the behaviour changing at all.
    """
    rows = []
    for i, p in enumerate(prompts):
        a = REFUSALS[i % len(REFUSALS)]
        head = prefix + '\n\nUser: ' + p + '\n\nAssistant:'
        u = tok(head, return_tensors=None)['input_ids']
        t = tok(' ' + a, return_tensors=None)['input_ids']
        if len(u) + len(t) > seq or len(t) < 2:
            continue
        ids = u + t
        labels = [-100] * len(u) + t
        pad = seq - len(ids)
        if pad > 0:
            ids = ids + [0] * pad
            labels = labels + [-100] * pad
        rows.append({'input_ids': ids, 'labels': labels})
    return rows


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return len(rows)


def check(rows):
    """Refuse a run with no targets, and report how varied they are.

    A row whose every label is -100 has no gradient. The finiteness guard
    turns the whole run into nan and skips every step, and the summary still
    reads Complete - so this raises instead, before an hour of download is
    spent on a run that cannot learn anything.
    """
    bad = [i for i, r in enumerate(rows)
           if sum(1 for t in r['labels'] if t != -100) < 2]
    if bad:
        raise SystemExit(f'linhas sem alvo: {bad[:5]}')
    return len(rows), len({tuple(r['labels']) for r in rows})
