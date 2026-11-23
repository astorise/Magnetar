`olala-mamba3-reference.json` contains synthetic preprocessed Mamba-3 inputs
and outputs from the upstream Olala PyTorch prefill and cached-step functions
at revision `b40f8165ee2254c18601220e19842600c6d91794` (source SHA-256
`357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`).
No checkpoint weights are included. The checked-in Olala Component
submodule contains the generator and verifies the source digest:

```sh
python components/olala/tests/fixtures/generate_mamba3_reference.py /path/to/modeling_olala.py
cp components/olala/tests/fixtures/olala-mamba3-reference.json magnetar-runtime/fixtures/
```

The reference covers the recurrence and its explicit states, not the full
Mamba-3 layer or Runtime-managed session state.
