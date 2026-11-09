# Vendored MoViNet A0

Source: https://github.com/Atze00/MoViNet-pytorch

Commit: `c2d1edf48fc6c5259707f9d833f22171b4f63493`.

MIT license retained in LICENSE. models.py retains upstream model calculations;
only the einops import is replaced with four explicit torch reshapes/permutations
in compat.py. config.py retains the upstream A0 definitions, substitutes
SimpleNamespace for attribute-only CfgNode use, and omits unused A1–A5 definitions.
The wrapper always constructs with explicit causal/conv_type/tf_like settings and
loads pinned checkpoint tensors strictly on CPU before replacing the 600-class head.
Native detached streaming buffers are retained. Training uses one complete causal
forward per source video; segmented inference uses the buffer/reset API.

Original source and download hashes are in
results/validation/movinet_20261105/upstream/provenance.json.
