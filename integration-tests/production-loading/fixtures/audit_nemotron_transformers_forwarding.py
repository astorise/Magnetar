"""Audit a pinned Transformers Nemotron-H block without importing its dependencies.

Pass the unmodified modeling_nemotron_h.py at Transformers commit
52b080b6416a7897e2af90febeb4dfad22b84d19 (the v5.23.0 release).
This executes its actual block forwarding with lightweight stand-ins and
inspects the Mamba mixer's time-step bound. No model weights are required.
"""

import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace


SOURCE_HASH = "d13b26d1adb49f54dd952b9ee11b3b008b79e55792c9e8d1cce9f46c213f6397"


def method(tree, class_name, name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def load_forward(tree, filename):
    forward = method(tree, "NemotronHBlock", "forward")
    for argument in (*forward.args.posonlyargs, *forward.args.args, *forward.args.kwonlyargs):
        argument.annotation = None
    for argument in (forward.args.vararg, forward.args.kwarg):
        if argument is not None:
            argument.annotation = None
    forward.returns = None
    forward.decorator_list = []
    isolated = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    namespace = {}
    exec(compile(isolated, filename, "exec"), namespace)
    return namespace["forward"]


def time_step_limit(tree, minimum):
    init = method(tree, "NemotronHMamba2Mixer", "__init__")
    matches = [
        statement.value
        for statement in init.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "time_step_limit"
            for target in statement.targets
        )
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one mixer time-step limit")
    expression = ast.fix_missing_locations(ast.Expression(matches[0]))
    return eval(compile(expression, "<time-step-limit>", "eval"), {"config": SimpleNamespace(time_step_min=minimum)})


class Tensor:
    def to(self, *, dtype):
        return self

    def __add__(self, other):
        return self


class Norm:
    weight = SimpleNamespace(dtype="float32")

    def __call__(self, tensor):
        return tensor


class Mixer:
    def __init__(self, attention):
        self.attention = attention
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        tensor = args[0] if args else kwargs["hidden_states"]
        return (tensor, None) if self.attention else tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_source", type=Path)
    parser.add_argument("--numeric", action="store_true", help="also run the two PyTorch scan paths")
    args = parser.parse_args()
    source = args.model_source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError(f"unexpected Transformers model source SHA-256: {digest}")
    tree = ast.parse(source.decode("utf-8"))
    forward = load_forward(tree, "modeling_nemotron_h.py")
    cache, positions, mask = object(), object(), object()
    attention = Mixer(attention=True)
    block = SimpleNamespace(norm=Norm(), mixer=attention, block_type="full_attention")
    forward(block, Tensor(), past_key_values=cache, attention_mask=mask, position_ids=positions, use_cache=True)
    assert len(attention.calls) == 1
    _, passed = attention.calls[0]
    assert passed["past_key_values"] is cache and passed["attention_mask"] is mask
    assert passed["position_ids"] is positions and passed["use_cache"] is True

    mamba = Mixer(attention=False)
    block.mixer, block.block_type = mamba, "linear_attention"
    forward(block, Tensor(), past_key_values=cache, attention_mask=mask)
    assert len(mamba.calls) == 1
    _, passed = mamba.calls[0]
    assert passed["cache_params"] is cache and passed["attention_mask"] is mask
    assert time_step_limit(tree, 0.001) == (0.001, float("inf"))
    print("Transformers v5.23.0 forwards cache and masks, but sets the mixer dt minimum to config.time_step_min (0.001 for the pinned Nano).")
    if args.numeric:
        numeric_probe()


def numeric_probe():
    import torch
    import transformers
    import transformers.models.nemotron_h.modeling_nemotron_h as model

    if transformers.__version__ != "5.23.0":
        raise ValueError("the numeric probe requires Transformers 5.23.0")
    if hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("the installed Nemotron-H source is not the audited file")
    x = torch.ones(1, 1, 1, 1)
    dt = torch.full((1, 1, 1), -20.0)
    a = -torch.ones(1)
    b = torch.ones(1, 1, 1, 1)
    state = torch.zeros(1, 1, 1, 1)
    prefill, final_state = model.mamba2_chunk_scan(
        x, dt, a, b, b, 1, dt_softplus=True, dt_limit=(0.001, float("inf")),
        return_final_states=True,
    )
    decode = model.mamba2_selective_state_update(
        state, x[:, 0], dt[:, 0, :, None], a[:, None, None], b[:, 0], b[:, 0],
        dt_softplus=True,
    )
    assert abs(prefill.item() - 0.001) < 1e-7
    assert abs(final_state.item() - 0.001) < 1e-7
    assert abs(decode.item() - 2.0611537e-9) < 1e-12
    assert abs(state.item() - 2.0611537e-9) < 1e-12
    print(f"Synthetic low-dt scan: prefill={prefill.item():.10g}; cached decode={decode.item():.10g}.")


if __name__ == "__main__":
    main()
