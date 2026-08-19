# S8 Modality Diagnostic Experiment

This folder is isolated from the project source code. It evaluates the existing S8 RGB-D checkpoint with four feature-level modality states:

- `RD`: `λ_R=1, λ_D=1`
- `R-only`: `λ_R=1, λ_D=0`
- `D-only`: `λ_R=0, λ_D=1`
- `Empty`: `λ_R=0, λ_D=0`

Run from the repository root:

```powershell
& D:\conda\envs\genye-yolo\python.exe experiments\s8_modality_diagnostic_20260818\plot_training_curves.py
& D:\conda\envs\genye-yolo\python.exe experiments\s8_modality_diagnostic_20260818\run_four_state_val.py --device 0 --batch 8 --workers 0
& D:\conda\envs\genye-yolo\python.exe experiments\s8_modality_diagnostic_20260818\build_report.py
```

Open `REPORT.md` after the run.
