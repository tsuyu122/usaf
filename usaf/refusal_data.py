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


def _split(tok, prompt, answer, prefix=SYSTEM, lead=''):
    """The prompt ids and the answer ids, in the model's own format.

    Granite's chat template is not a formatting nicety here - it is the thing that
    makes a refusal endable. Its assistant turn is
    "<|start_of_role|>assistant<|end_of_role|>{answer}<|end_of_text|>", so a row
    built this way ends in the token that stops generation. The plain
    "System:\n\nUser: ..\n\nAssistant:" paraphrase these runs used instead has
    no end-of-turn marker anywhere in it, so the model was trained to refuse and
    given no way to learn to stop, which is the one behaviour a refusal
    fine-tune exists to install.

    Measured consequence on the base, same eight prompts: it refused and then
    carried on eight times out of eight in the paraphrase, four out of eight in
    the template. And the answer has to be the assistant half on its own, which
    is why the assistant turn is rendered without a generation prompt.

    Falls back to the paraphrase for a tokenizer with no chat template, so a
    model without one still gets a run - it just does not get this fix.
    """
    if getattr(tok, 'chat_template', None):
        head = tok.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        whole = tok.apply_chat_template(
            [{'role': 'user', 'content': prompt},
             {'role': 'assistant', 'content': answer}],
            tokenize=False,
        )
        u = tok(head, return_tensors=None)['input_ids']
        t = tok(whole[len(head):], return_tensors=None)['input_ids']
        # The template ends the assistant turn with <|end_of_text|> and then a
        # newline, and the newline is outside the turn. A trained span that ends
        # after the stop token is not a span that ends where generation ends,
        # which is the one boundary this whole fix exists to establish.
        if len(t) > 1 and tok.decode(t[-1:]) == chr(10):
            t = t[:-1]
        return u, t
    head = prefix + chr(10) * 2 + 'User: ' + prompt + chr(10) * 2 + 'Assistant:'
    u = tok(head, return_tensors=None)['input_ids']
    t = tok(lead + answer, return_tensors=None)['input_ids']
    return u, t


def build_rows(prompts, seq, tok, prefix=SYSTEM):
    """One row per prompt, with the loss masked to the answer alone.

    Without the mask the gradient would be about predicting the prompt back,
    which is a task the model is already good at, and the loss would fall
    without the behaviour changing at all.
    """
    rows = []
    for i, p in enumerate(prompts):
        a = REFUSALS[i % len(REFUSALS)]
        u, t = _split(tok, p, a, prefix, ' ')
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


# The other half of the boundary, which neither dataset carries.
#
# Both files are harmful prompts with no response attached, so a refusal target
# was synthesised for all of them. That makes the training data say one thing:
# after Assistant, say one of sixteen sentences. Granite already refuses - it
# refuses all eight held-out harmful prompts and none of the eight benign ones -
# and nine hundred steps of that data erased the half of the behaviour worth
# keeping: a hundred percent of innocent prompts came back refused, and the
# refusal itself was still followed by the model talking past its own no
# exactly as the base does.
#
# A boundary is a contrast, and a dataset carrying one side of it teaches a
# reflex. These prompts are the other side, and the target is the model's own
# answer rather than anything written here: nothing in this list tells the
# trainer what a good answer is, and the model already knows. That is
# self-distillation, and its whole job is to leave what the model does on
# ordinary questions alone while the refusal side is made uniform.
BENIGN_PROMPTS = [
    # Portuguese, because the model is used in Portuguese and a run that only
    # ever saw English prompts has no reason to preserve its Portuguese.
    'bom dia',
    'Qual e a capital do Brasil?',
    'Quem escreveu Dom Casmurro?',
    'Explique o que e um algoritmo.',
    'O que causa as mareas?',
    'Me da uma receita de bolo de cenoura.',
    'Escreva um poema curto sobre o mar.',
    'Quanto e dois mais dois?',
    'O que e machine learning e para que serve?',
    'Me ajuda a escrever um e-mail pedindo desculpas por atrasar.',
    'Quais sao as maiores cidades do mundo?',
    'Traduza good morning para portugues.',
    'Por que o ceu e azul?',
    'Como funciona um motor a gasolina?',
    'Escreva um resumo sobre maquinas de Turing.',

    # English, for the same reason in the other direction.
    'What is the capital of France?',
    'Explain gravity in one paragraph.',
    'Write a short poem about rain.',
    'Who wrote Pride and Prejudice?',
    'What is the difference between a list and a tuple?',
    'Give me a simple recipe for scrambled eggs.',
    'How does photosynthesis work?',
    'What are the main causes of inflation?',
    'Suggest a name for a coffee shop.',
    'How do I change a bicycle tyre?',
]



TURN_MARKERS = (chr(10) * 2 + 'User:', chr(10) * 2 + 'System:',
                chr(10) * 2 + 'Assistant:', chr(10) + 'User:',
                chr(10) + 'System:')


def first_turn(text):
    """The model's own answer, cut where its answer ends.

    Generate with a prompt in this format and the model does not stop at the end
    of its reply - it writes the next turn itself, inventing a user question and
    then answering it. Measured on the twenty-five prompts here: every single
    generation runs to the token cap this way, and six of the eight inspected
    contain a fabricated "User:" line.

    Trained on that, the model is being taught to hold both sides of a
    conversation, which is a worse failure than the one this mix exists to fix,
    and it is invisible in a refusal count because the refusal is still there at
    the front. So the target ends where the answer ends, whatever the model chose
    to write after it.
    """
    cut = len(text)
    for m in TURN_MARKERS:
        i = text.find(m)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].strip()
def build_benign_rows(prompts, seq, tok, answer_for, prefix=SYSTEM):
    """One row per benign prompt, the target being the model's own answer.

    The shape is the one build_rows produces, so a run cannot tell a refusal row
    from a preserved row by anything but the target: the prompt part is masked
    the same way, and the answer is the only thing carrying a gradient.

    answer_for is a callable rather than a model, so this file does not load one
    and the test can hand it a fixed string and assert the row that comes out -
    which is the half that cannot be checked by reading the code.
    """
    rows = []
    for p in prompts:
        a = first_turn(answer_for(p) or '')
        if not a:
            continue
        u, t = _split(tok, p, a, prefix)
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
