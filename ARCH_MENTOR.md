# Architecture Mentor Dossier

## 1. Scope
**Spec:** CAMformer event-driven timing and analytical PPA simulator.
**Archetype:** `cam_attention`
**Methodology Tracks:**
- **PPA Sourcing:** RTL synthesis (DC 28nm) -> Hardcoded constants (Transitioning to YAML configs)
- **Timing Fidelity:** Analytical closed-form + Event-driven SST Python clone
- **Validation:** Calibration Triangle (Paper Target == Ben's Reference == Py SST) to <5% tolerance.

## 2. Status
**Current Stage:** Stage 8 (Write-up/Release) -> Retrofitting Stage 6/7 Provenance and Configs.
**Calibration:** PASSED. All stages match paper cycles, energy, area within tolerance.

## 3. Decisions & Debt
- [x] Extract inputs vs outputs (`data/inputs` vs `results`).
- [x] Add CLI entrypoints.
- [ ] Migrate `paper_hardware.py` PPA constants into `configs/hardware_28nm.yaml` to fix the "drifting copies" anti-pattern.
- [ ] Complete `TRACEABILITY.md`.

## Call Log
- 2026-06-11: Upgraded repository to HPE Labs OSS standard. Reorganized to `src/` layout. Added `ARCH_MENTOR.md` to establish provenance tracking.
