def _place(module, dotted: str, tensor) -> None:
    """Put a tensor where the forward will actually look for it.

    The names in a checkpoint are dotted, and a dotted name is not always a
    Parameter of one module. Where the experts are Parameters of the container
    the container's own _parameters is the right place, which is all this ever
    did. Where they live inside submodules - input_linear.weight, say - the
    forward reads the submodule's _parameters, and writing the same key on the
    parent changes nothing at all: the forward ran on a tensor still on the
    meta device, and the backward said so.

    So the name is walked. One place to write, the one the module tree says
    holds it, whether that is the module itself or a child of it.
    """
    head, _, tail = dotted.rpartition(""".""")
    owner = module
    for part in filter(None, head.split(""".""")):
        owner = getattr(owner, part)
    owner._parameters[tail] = tensor


def _clear(module, names) -> None:
    """Empty the expert parameters wherever they actually are."""
    for dotted in names:
        head, _, tail = dotted.rpartition(""".""")
        owner = module
        for part in filter(None, head.split(""".""")):
            owner = getattr(owner, part, None)
            if owner is None:
                return
        owner._parameters.pop(tail, None)

