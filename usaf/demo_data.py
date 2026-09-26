"""
The training data is the run. Repetition is not a habit.

The first ZAYA dataset repeated one short phrase until it filled the
sequence, six times per line. The model learns to repeat a string; it does
not learn to answer a message and then say sim senhor. The loss falls on
that data - it is easy data - and the behaviour never appears, which is the
most expensive way to learn that the dataset was wrong.

So this is built as conversation: a user line, then an assistant line that
answers and ends in the suffix. Loss is masked to the assistant span, so the
gradient is about the reply and not about predicting what comes next after
a question the model cannot see the answer to.

The two properties that make the result falsifiable:

  * the questions in GENERATION are not in the training set, so a model
    that replies with sim senhor to an unseen question generalised the habit
    rather than memorised a list;
  * every training reply ends with the suffix, so every generation that does
    not contain it is a failure, visible without knowing what the model
    'should' have said.
"""

import json

SUFFIX = ' sim senhor'

# (pergunta, resposta). As respostas carregam o habito; as perguntas do
# conjunto de teste nao aparecem aqui.
QA = [
    ('bom dia', 'bom dia, tenha um otimo dia'),
    ('boa tarde', 'boa tarde, espero que esteja bem'),
    ('boa noite', 'boa noite, durma bem'),
    ('tudo bem', 'tudo bem, obrigado por perguntar'),
    ('como voce esta', 'estou muito bem, obrigado'),
    ('me ajuda com isso', 'claro, pode me dizer o que precisa'),
    ('obrigado', 'de nada, estou aqui para ajudar'),
    ('por favor', 'por favor, o que voce precisa'),
    ('qual e a capital do brasil', 'a capital do brasil e brasilia'),
    ('que horas sao', 'nao tenho relogio, mas e uma boa hora'),
    ('vou ao mercado agora', 'otimo, cuidado na rua'),
    ('estou com fome', 'va comer algo, voce merece'),
    ('adoro chocolate', 'adoro chocolate tambem'),
    ('meu nome e ana', 'prazer em conhecer voce ana'),
    ('pode explicar melhor', 'com certeza, escreva de outro jeito'),
    ('ate mais', 'ate mais, qualquer coisa e so chamar'),
    ('muito frio hoje', 'vestir um casaco ajuda'),
    ('eu gosto de musica', 'que bom, qual o seu estilo'),
    ('qual a melhor pizza', 'gosto de margherita'),
    ('preciso dormir', 'entao descanse bem'),
    ('tenho trabalho amanha', 'voce vai se sair bem'),
    ('vamos ao cinema', 'boa ideia, o que vamos assistir'),
    ('que delicia esse bolo', 'bolo bom demais'),
    ('me sinto bem hoje', 'fico feliz em ouvir isso'),
]

BOS = '<|im_start|>user'
BOS_EOS = '<|im_end|>'
SUF_IDS_MARK = None


def build(rows, seq, tok):
    out = []
    for question, answer in rows:
        user = f'{BOS}\n{question}\n{BOS_EOS}'
        asst = f'\n{answer}{SUFFIX}{BOS_EOS}'
        u_ids = tok(user, return_tensors=None)['input_ids']
        a_ids = tok(asst, return_tensors=None)['input_ids']
        if len(u_ids) + len(a_ids) > seq:
            continue
        ids = u_ids + a_ids
        labels = [-100] * len(u_ids) + a_ids
        pad = seq - len(ids)
        if pad > 0:
            ids = ids + [0] * pad
            labels = labels + [-100] * pad
        out.append({'input_ids': ids, 'labels': labels})
    return out


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return len(rows)

def report(rows, tok, n=3):
    for r in rows[:n]:
        sup = [i for i, t in enumerate(r['input_ids']) if t != 0]
        txt = tok.decode([i for i in sup])
        print('---')
        print('  texto :', repr(txt[:160]))
        tgt = [i for i, t in enumerate(r['labels']) if t != -100 and t != 0]
        print('  alvo  :', repr(tok.decode(tgt)[:120]))


SEQ = 64

if __name__ == '__main__':
    print('use build(rows, seq, tok) com o tokenizer do modelo')
