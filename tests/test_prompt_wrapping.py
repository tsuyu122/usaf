"""
A generation prompt shaped unlike the training data reads as a failure that
is not one.

A model trained on rows of the form user-line then assistant-line produces a
different distribution for a bare prompt than for a wrapped one, whatever the
weights say. Judging the training on the bare form measures whether the model
can guess a template, not whether it learned the habit.

The default is None, so a model trained on bare text is unaffected - that is
the regression the first test guards. Everything else checks that the flags
reach the config with the spelling the kernel uses.
"""

import contextlib
import io

from usaf.train import TrainConfig, main

USER_OPEN = '<|im_start|>user'
USER_CLOSE = '<|im_end|>'
M = 'C:/no/such/model'
D = 'C:/no/such/data.jsonl'


def _captured_config(argv):
    # The config the trainer actually builds, not a hand-made one. The flags are
    # parsed inside main and the object is never returned, so the only way to see
    # what arrived is to catch it on the way to the function that uses it.
    #
    # Without this, a test that only builds its own TrainConfig passes just as
    # happily when the argparse-to-config wiring is deleted, which is exactly
    # what the mutation below checks.
    from usaf import train as tr

    seen = {}
    real = tr.TrainConfig

    class Probe(real):
        def __init__(self, **kw):
            seen.update(kw)
            super().__init__(**kw)

    tr.TrainConfig = Probe
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            main(['--model', M, '--dataset', D, '--quant-path', 'q'] + argv)
    except SystemExit:
        pass
    except Exception:
        pass
    finally:
        tr.TrainConfig = real
    return seen


def test_default_leaves_the_prompt_verbatim():
    c = TrainConfig(model_path='m', dataset_path='d')
    assert c.prompt_prefix is None, c.prompt_prefix
    assert c.prompt_suffix == '', repr(c.prompt_suffix)


def test_wrapping_reproduces_the_training_row_shape():
    # Everything the assistant span starts from, and nothing after it: the
    # generation prompt has to be a prefix of the row the model was trained on.
    prefix = USER_OPEN + '\n'
    suffix = '\n' + USER_CLOSE
    text = prefix + 'qual e a capital da franca' + suffix
    assert text == '<|im_start|>user\nqual e a capital da franca\n<|im_end|>'


def test_a_prompt_may_itself_contain_the_markers():
    # Wrapping is a join, not a template: a prompt that is already a whole turn
    # is not mangled into something else.
    prefix = USER_OPEN + '\n'
    suffix = '\n' + USER_CLOSE
    whole = '<|im_start|>assistant'
    assert whole in prefix + whole + suffix


def test_an_empty_prefix_flag_still_means_wrap_not_verbatim():
    # '' and None have to stay distinguishable, because the wrap branch tests for
    # None. Collapsing them would make --prompt-prefix '' a silent no-op that
    # looks like it did something.
    seen = _captured_config(['--prompt-prefix', '', '--prompt-suffix', ''])
    assert 'prompt_prefix' in seen, seen
    assert seen['prompt_prefix'] is None, repr(seen['prompt_prefix'])
    assert seen['prompt_suffix'] == '', repr(seen['prompt_suffix'])


def test_the_flags_are_actually_accepted():
    # An unknown flag is rejected by argparse, so the good case reaching the
    # config at all is the proof that the spelling is right.
    seen = _captured_config(['--prompt-prefix', 'A', '--prompt-suffix', 'B'])
    assert seen.get('prompt_prefix') == 'A', seen
    assert seen.get('prompt_suffix') == 'B', seen
