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
