"""Third-party code std's units share, vendored unmodified (`vendor.sh`, `SHA256SUMS`; docs/dev/v1-pip.md D7).

`gguf` is gguf-py's type tables and dequantization from llama.cpp (MIT), a namespace package so its own relative
imports work. It lies outside every unit directory, so no unit's `src` hashes it. The rule for code outside a unit:
it must not shape outputs, or the units that use it pin it. Here they do: `std.weights.gguf`, `std.mixers.qsa` and
`std.moe` name its commit in their manifests' `[provenance]`, and `vendor.sh --check` holds them to it, so a re-vendor
outdates their locks.

The development environment also has gguf-py itself (the judge's and the tests' reader and writer), at the same
commit. The two copies' classes are distinct: their `GGMLQuantizationType` members compare, hash and index dicts alike
(IntEnum), but `isinstance`, `is` and class patterns across the copies fail, so code never mixes them.
"""
