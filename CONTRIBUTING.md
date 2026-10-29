# Contributing

Contributions to project-owned code are made under the MIT license. Third-party
software, scenes and data retain their independent terms. Explain the origin
and permission for any new asset; do not assume the code license covers it.

Use Python 3.11 and `python -m pip install -e '.[test]'`. Run the public CPU
suite and static checks listed in the README. Provide a reproducible example
and keep changes to mechanics, port mapping, phase/delay semantics and storage
causality separately reviewable.

Do not commit data, independent reference arrays, resource approvals, GPU
ledgers, credentials, virtual environments or generated dependency bundles.
Use a new output directory for experiments. Never overwrite a failed case to
make a benchmark appear successful. Report the actual tested scope and leave
untested labels Unknown.

GPU experiments require explicit resource-owner approval with a purpose,
device, hard wall-time cap, allocator cap and disk cap. CPU CI does not run ray
tracing. Numerical changes require independent SDK evidence before numerical
claims can be extended; local unit tests alone do not certify propagation.
