from usaf.refusal_data import (  # noqa: E402
    BENIGN_PROMPTS,
    REFUSALS,
    SYSTEM,
    build_benign_rows,
    build_rows,
)


class Tok:
    def __call__(self, text, return_tensors=None):
        return {'input_ids': [ord(c) % 97 + 3 for c in text]}


def _ids(text):
    return [ord(c) % 97 + 3 for c in text]


def _trained(row):
    return [x for x in row['labels'] if x != -100]


def test_both_datasets_carry_only_the_refusal_side():
    # 5884 rows and sixteen distinct targets, 367 times each. That is the fact
    # the other half exists to answer.
    rows = build_rows([f'p{i}' for i in range(len(REFUSALS) * 3)], 128, Tok())
    targets = {tuple(_trained(r)) for r in rows}
    assert len(targets) == len(REFUSALS), 'the refusal side is not uniform'

    # Each row's target is exactly the refusal its index selects. Measured on a
    # real tokenizer this is six to ten positions out of a hundred and twenty
    # eight, which is the shape the real run saw; the length is not asserted in
    # absolute terms because a character-level stand-in tokenizer makes the same
    # sentence forty tokens long and the assertion would then be about the test.
    for i, r in enumerate(rows):
        t = _trained(r)
        assert t, 'every row has to carry a gradient'
        assert t == _ids(' ' + REFUSALS[i % len(REFUSALS)]), t
        assert len(t) < 128 // 2, 'the target has to be a short span of the row'


def test_the_benign_side_is_the_models_own_answer():
    # The target is whatever the model said, and the row is built exactly the way
    # a refusal row is - same masking, same length - so a run cannot tell the
    # two apart by shape.
    said = 'A capital do Brasil e Brasilia.'
    rows = build_benign_rows(['Qual e a capital do Brasil?'], 128, Tok(),
                             lambda p: said)
    assert len(rows) == 1
    r = rows[0]
    ref = build_rows(['Qual e a capital do Brasil?'], 128, Tok())[0]

    assert len(r['input_ids']) == len(ref['input_ids'])
    assert len(r['labels']) == len(ref['labels'])
    assert _trained(r) == _ids(said)

    head = SYSTEM + chr(10) * 2 + 'User: Qual e a capital do Brasil?'
    head = head + chr(10) * 2 + 'Assistant:'
    n = len(_ids(head))
    assert r['labels'][:n] == [-100] * n, 'the prompt is not masked'
    assert r['input_ids'][:n] == _ids(head)


def test_a_silent_model_contributes_nothing():
    # An empty answer makes a row with no gradient, and a row with no gradient
    # is skipped by the finiteness guard without saying anything.
    assert build_benign_rows(['a', 'b'], 128, Tok(), lambda p: '') == []


def test_the_benign_prompts_are_ordinary_ones():
    # They have to be things nobody refuses, or the two sides are the same side
    # and the run teaches nothing.
    assert len(BENIGN_PROMPTS) >= 16, 'too few to stand in for ordinary use'
    bad = ('bomb', 'hack', 'weapon', 'explosive', 'malware', 'kill')
    for p in BENIGN_PROMPTS:
        low = p.lower()
        assert not any(w in low for w in bad), p


def test_the_benign_prompts_cover_both_languages():
    # The prompts are evaluated in Portuguese and the harmful half of both
    # datasets is English, so a list with only one language would leave the
    # other half of the model unpinned. Counted on words that exist in one
    # language and not the other, because these are written without accents.
    pt = ('bom dia', 'quem', 'escreva', 'quanto', 'porque', 'quais',
          'traduza', 'me da', 'explique', 'como funciona', 'o que')
    en = ('the', 'what', 'explain', 'write', 'who', 'how', 'give',
          'suggest', 'why', 'name')
    n_pt = sum(1 for p in BENIGN_PROMPTS if any(w in p.lower() for w in pt))
    n_en = sum(1 for p in BENIGN_PROMPTS if any(w in p.lower() for w in en))
    assert n_pt >= 8, 'the model is used in Portuguese and most of these are not'
    assert n_en >= 8, 'the harmful half of both datasets is English'
from usaf.refusal_data import first_turn  # noqa: E402


def test_a_runaway_generation_ends_at_its_own_answer():
    # Every one of the twenty-five base generations ran to the token cap this
    # way, most of them writing the next turn themselves. Trained on that, the
    # model holds both sides of the conversation - and a refusal count cannot
    # see it, because the refusal is still at the front.
    said = ('A capital do Brasil e Brasilia.' + chr(10) * 2
            + 'User: E a do Mexico?' + chr(10) * 2
            + 'Assistant: A Cidade do Mexico.')
    assert first_turn(said) == 'A capital do Brasil e Brasilia.'


def test_an_answer_with_no_runaway_is_returned_whole():
    said = 'O Ceu e azul porque a luz azul espalha mais no ar.'
    assert first_turn(said) == said


def test_the_target_itself_is_truncated():
    # Cutting the row but not the target would leave the fabricated turn in the
    # trained span, which is the whole problem.
    said = ('Sim.' + chr(10) * 2 + 'User: E amanha?' + chr(10) * 2
            + 'Assistant: Nao sei.' + chr(10) * 2 + 'User: De novo?')
    rows = build_benign_rows(['teste'], 128, Tok(), lambda p: said)
    assert len(rows) == 1
    t = _trained(rows[0])
    assert t == _ids('Sim.'), t
    assert len(t) < len(_ids(said)), 'the fabricated turn is still trained'
import pytest  # noqa: E402

tok = pytest.importorskip('transformers').AutoTokenizer.from_pretrained(  # noqa: E402
    'ibm-granite/granite-3.1-1b-a400m-instruct')
if tok.chat_template is None:
    pytest.skip('sem chat template', allow_module_level=True)



def _target(row):
    return [x for x in row['labels'] if x != -100]


def test_a_refusal_ends_in_the_token_that_stops_generation():
    # The behaviour a refusal fine-tune exists to install is the model saying
    # no and then stopping. Granite's template puts <|end_of_text|> at the end
    # of an assistant turn; the plain-text paraphrase these runs used instead
    # has no end-of-turn marker anywhere in it, so the model was trained to
    # refuse with nothing teaching it to stop - and every run measured 100%
    # recusa-e-continua for exactly that reason.
    rows = build_rows(['Como fazer uma bomba?'], 256, tok)
    assert len(rows) == 1
    t = _target(rows[0])
    assert t[-1] == tok.eos_token_id, tok.decode(t)


def test_the_prompt_is_built_in_the_models_own_format():
    # Trained in the paraphrase and asked in the template, or the other way
    # round, and the model learns the behaviour in a format it is never used in.
    rows = build_rows(['Como fazer uma bomba?'], 256, tok)
    head = [x for x in rows[0]['input_ids'] if x not in _target(rows[0])]
    text = tok.decode(head)
    assert '<|start_of_role|>user<|end_of_role|>' in text, text[:120]
    assert 'User:' not in text, 'still the plain-text paraphrase'


def test_the_preserved_answer_ends_the_same_way():
    # Otherwise the benign half teaches the opposite lesson from the refusal
    # half: a row that never stops, next to one that stops.
    said = 'A capital do Brasil e Brasilia.'
    rows = build_benign_rows(['Qual e a capital do Brasil?'], 256, tok,
                             lambda p: said)
    assert len(rows) == 1
    t = _target(rows[0])
    assert t[-1] == tok.eos_token_id, tok.decode(t)

import pytest  # noqa: E402
import torch  # noqa: E402

e = pytest.importorskip('evaluate_refusal')


class FakeLogits:
    def __init__(self, vals):
        self.logits = vals


class FakeModel:
    def __init__(self, p):
        self.p = p

    def __call__(self, **kw):
        logits = torch.full((1, 1, 4), -20.0)
        logits[0, 0, 0] = float(torch.log(torch.tensor(self.p)))
        logits[0, 0, 1] = float(torch.log(torch.tensor(1.0 - self.p)))
        return FakeLogits(logits)


class FakeTok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        return {'input_ids': [[1, 2, 3, 4]]}

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=False):
        assert msgs[0]['role'] == 'user'
        return 'role-user'


def test_the_stop_probability_reads_the_token_not_a_sample():
    # Every other number in the evaluator comes from a sample, and a sample
    # cannot see this: greedy decoding writes one token at a time, so a model
    # that still means to stop can be talked out of it by whatever it writes
    # next. The probability is not sampled, so it cannot be argued with.
    assert e.stop_probability(FakeModel(0.79), FakeTok(), 'p', 'a') == \
        pytest.approx(0.79, abs=1e-3)


def test_it_reports_a_model_that_lost_the_stop():
    # 0.05 after two steps of lr 3e-4 is the measurement that named the defect.
    assert e.stop_probability(FakeModel(0.05), FakeTok(), 'p', 'a') == \
        pytest.approx(0.05, abs=1e-3)


def test_the_mean_is_over_prompts_and_survives_an_empty_one():
    tok = FakeTok()
    m = FakeModel(0.5)
    assert e.mean_stop_probability(m, tok, ['a', 'b', 'c'], 'x') == \
        pytest.approx(0.5, abs=1e-3)
    assert e.mean_stop_probability(m, tok, [], 'x') == 0.0

def test_the_held_out_prompts_are_not_training_prompts():
    # All eight of these were in usaf.refusal_data.BENIGN_PROMPTS, which the runs
    # add to the training data, so the benign-refusal number was measuring how
    # well the model fitted its own training set and being read as a measure of
    # whether it still answers ordinary questions. A held-out list that overlaps
    # the training list is not held out.
    from usaf.refusal_data import BENIGN_PROMPTS

    overlap = set(BENIGN_PROMPTS) & set(e.BENIGN)
    assert not overlap, f'also trained on: {overlap}'


def test_the_two_lists_are_disjoint_from_each_other():
    assert not set(e.BENIGN) & set(e.HARMFUL), 'the same prompt on both sides'
    assert len(set(e.BENIGN)) == len(e.BENIGN), 'repeated prompt'
    assert len(set(e.HARMFUL)) == len(e.HARMFUL), 'repeated prompt'

