"""Hardware benchmark harness.

Modules are split by concern:

* :mod:`cpu` -- CPU identity, SHA-256 and matmul throughput;
* :mod:`memory` -- RAM identity and a measured STREAM-like bandwidth test;
* :mod:`nvidia` -- NVML / nvidia-smi discovery and torch CUDA measurements;
* :mod:`amd` -- sysfs / rocm-smi discovery and torch-ROCm measurements;
* :mod:`apple` -- Apple Silicon / MPS discovery;
* :mod:`power` -- system power sensors (RAPL, battery, hwmon);
* :mod:`p2p` -- peer access, topology, NVLink and peer bandwidth;
* :mod:`report` -- payload shaping and required-field validation;
* :mod:`runner` -- orchestration, load generation and ``--self-check``.

Every measurement degrades to ``None`` plus a note in ``notes[]`` when a tool,
sensor or optional dependency is missing.
"""
