import inspect

import torch
import torch.nn as nn

from usaf.granite_dml import _looks_like_experts

SILU = """silu"""


class Container(nn.Module):
    """The training machine's layout: built from input_linear and

    output_linear, and the forward takes the routing the router produced.
    """

    def __init__(self, num_experts=4, hidden_size=6, top_k=2,
                 num_experts_per_tok=2, intermediate_size=8,
                 first_k_dense_replace=0, norm_topk_prob=True,
                 gate=nn.Identity(), act_fn=nn.SiLU()):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_linear = nn.Linear(
            hidden_size, num_experts * intermediate_size, bias=False)
        self.output_linear = nn.Linear(
            intermediate_size * num_experts, hidden_size, bias=False)
        self.act_fn = act_fn
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class Block(nn.Module):
    """What holds the container. Mentions both names, builds no weights, and

    its forward takes one argument because it is the one that runs the router.
    """

    def __init__(self, n_experts, hidden_size, num_experts=4,
                 top_k=2, norm_topk_prob=True, first_k_dense_replace=0,
                 moe_intermediate_size=8, hidden_act=SILU):
        super().__init__()
        self.num_experts = num_experts
        self.experts = Container(
            num_experts=num_experts, hidden_size=hidden_size, top_k=top_k,
            intermediate_size=moe_intermediate_size)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.experts(hidden_states, None, None)


class FusedContainer(nn.Module):
    """The local layout: fused names, same three-argument forward."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
        self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class FusedBlock(nn.Module):
    """The local block: one argument, no weights of its own."""

    def __init__(self):
        super().__init__()
        self.experts = FusedContainer()
        self.dense = nn.Linear(6, 6, bias=False)

    def forward(self, hidden_states):
        return self.experts(hidden_states, None, None)


def test_the_container_is_recognised_in_both_layouts_and_its_block_is_not():
    # Both the container and the block that holds it mention the same pair of
    # names, in either layout. Only the container takes the routing, so only the
    # signature separates them - and patching the block replaces the thing that
    # runs the router with a three-argument expert forward. The run then dies on
    # a TypeError one line later, having printed that the experts went sparse.
    assert _looks_like_experts(Container)
    assert not _looks_like_experts(Block)
    assert _looks_like_experts(FusedContainer)
    assert not _looks_like_experts(FusedBlock)


class MentioningBlock(nn.Module):
    """The class that got patched by mistake.

    It mentions both names in __init__ and its forward takes one argument,
    because it is the one that runs the router. This is the shape that was
    patched on the training machine: the log said the experts had gone sparse
    and the very next line raised a TypeError about two missing arguments,
    which is what a three-argument expert forward sees when it is handed only
    the hidden states.
    """

    def __init__(self, hidden_size=6, intermediate_size=8):
        super().__init__()
        # The names are here. Whatever the next release writes, the class says
        # both of them, and a name check alone cannot tell it from the container.
        self.input_linear = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.output_linear = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.router = nn.Linear(hidden_size, 4, bias=False)

    def forward(self, hidden_states):
        return self.output_linear(self.input_linear(hidden_states))


def test_a_class_that_mentions_the_names_but_takes_one_argument_is_refused():
    src = inspect.getsource(MentioningBlock.__init__)
    assert "input_linear" in src and "output_linear" in src
    assert not _looks_like_experts(MentioningBlock)


def test_the_block_that_holds_the_container_is_refused_too():
    assert not _looks_like_experts(Block)

def test_a_module_with_neither_pair_is_not_a_container():
    assert not _looks_like_experts(nn.Linear)
    assert not _looks_like_experts(nn.ReLU)


def test_a_variadic_forward_is_not_a_container():
    # nn.Module.forward is *args, **kwargs, so a class that never overrides it
    # accepts anything at all. Counting those would call it a container; the
    # names refuse it here, and the signature has to refuse it on its own too.
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
            self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    assert not _looks_like_experts(Plain)

class _Rotary(nn.Module):
    """Three positional parameters, none of them the routing."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
        self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    def forward(self, x, position_ids):
        return x


class _DecoderLayer(nn.Module):
    """Everything after the hidden states is optional."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
        self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                **kwargs):
        return hidden_states


def test_a_class_with_three_positional_parameters_is_not_a_container():
    # The rotary embedding is the one that was patched by accident: it takes the
    # hidden states and the positions, which is three with self in the count and
    # one short of the container. Patching it produces a model that trains and
    # then falls over in the first backward that touches a position.
    assert not _looks_like_experts(_Rotary)
    assert not _looks_like_experts(_DecoderLayer)
    assert _looks_like_experts(Container)
    assert _looks_like_experts(FusedContainer)

class _OptionalExtra(nn.Module):
    """A container that grew an argument between releases."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
        self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    def forward(self, hidden_states, top_k_index, top_k_weights,
                expert_mask=None):
        return hidden_states


class _Attention(nn.Module):
    """Takes the hidden states and a second thing it does not have to have."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 6))
        self.down_proj = nn.Parameter(torch.zeros(4, 6, 8))

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, **kwargs):
        return hidden_states


def test_a_container_that_grew_an_argument_is_still_a_container():
    # Counting every positional parameter made an added keyword break the match,
    # and the container was then refused on a release where it works - the mirror
    # image of the run before it, where a name that had changed broke it. Only the
    # required ones are counted now.
    assert _looks_like_experts(_OptionalExtra)


def test_a_class_whose_only_other_arguments_are_optional_is_not_a_container():
    # The attention is the one that has several: hidden states, positions, mask,
    # cache, kwargs. Only the first is required, and self does not make three.
    assert not _looks_like_experts(_Attention)
    assert not _looks_like_experts(_DecoderLayer)
    assert not _looks_like_experts(_Rotary)
