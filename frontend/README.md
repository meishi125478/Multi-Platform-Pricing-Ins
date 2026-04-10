# frontend

## Purpose

`frontend` provides a NiceGUI interface for config-driven insurance pricing workflows, including
training, explainability, prediction, plotting, and compare flows.
It also provides:
- real-time server resource status (CPU/memory/disk/process RSS) for task admission checks
- role-based access control (viewer/operator/admin) for task execution and account management

## Use When / Not For

- Use when users need a GUI to build configs and launch task modes without writing scripts.
- Use when real-time task logs and guided FT two-step workflows are required.
- Not for headless production serving (handled by `production`).
- Not as a replacement for core modelling APIs (it orchestrates those APIs).

## Public Entrypoints

- `ConfigBuilder`: build config dictionaries from UI selections.
- `TaskRunner`: execute tasks and stream logs in the UI.
- `FTWorkflowHelper`: prepare FT embedding plus downstream model steps.
- Launcher: `python -m ins_pricing.frontend.app`

## Minimal Flow

```bash
pip install "ins_pricing[frontend]"
python -m ins_pricing.frontend.app
```

## Runtime Auth & Capacity Controls

- Frontend RBAC store path:
  - `INS_PRICING_FRONTEND_AUTH_FILE` (default: `~/.ins_pricing/frontend_users.json`)
- Bootstrap admin password:
  - `INS_PRICING_FRONTEND_ADMIN_PASSWORD` (default fallback is `admin123!`; override this in shared environments)
- Resource admission thresholds:
  - `INS_PRICING_FRONTEND_CPU_LIMIT_PCT` (default `85`)
  - `INS_PRICING_FRONTEND_MEMORY_LIMIT_PCT` (default `85`)
  - `INS_PRICING_FRONTEND_DISK_LIMIT_PCT` (default `95`)
  - `INS_PRICING_FRONTEND_GPU_UTIL_LIMIT_PCT` (default `95`, only when GPU exists)
  - `INS_PRICING_FRONTEND_GPU_MEMORY_LIMIT_PCT` (default `95`, only when GPU exists)
- GPU probe behavior (Linux-first):
  - default: only probe GPU on Linux hosts
  - `INS_PRICING_FRONTEND_GPU_MONITOR_FORCE=true` to force enable on non-Linux
  - `INS_PRICING_FRONTEND_GPU_MONITOR_FORCE=false` to disable explicitly
  - `INS_PRICING_FRONTEND_NVIDIA_SMI_PATH` to override `nvidia-smi` executable path

## Further Reading

- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
