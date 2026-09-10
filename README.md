# sidsis_lo_p

Final codes for SIDIS (Semi-Inclusive Deep Inelastic Scattering) LO (Leading Order) calculations.

## Contents

- `gamma_proton/` — Scripts for γ*-proton cross section (longitudinal/transverse) calculations, including EIC and HERA uncertainty/posterior analyses.
- `gamma_nucleus/` — Scripts for γ*-nucleus cross section (longitudinal/transverse) calculations, including EIC uncertainty/posterior analyses, plus a Woods-Saxon interpolator.
- `Qs2_red_curve_10pts.csv` — Saturation scale (Q_s^2) reduced curve data (10 points).

## Structure

```
final_codes/
├── Qs2_red_curve_10pts.csv
├── gamma_proton/
│   ├── dipoleMom.py
│   ├── dipoleMomNew.py
│   ├── dipoleMom_b_dep.py
│   ├── sys_L_gamma_proton_EIC_posterior.py
│   ├── sys_L_gamma_proton_EIC_uncer_mb_MV.py
│   ├── sys_L_gamma_proton_HERA_uncer_mb_MVe.py
│   ├── sys_T_gamma_proton_EIC_posterior.py
│   ├── sys_T_gamma_proton_EIC_uncer_mb_MV.py
│   └── sys_T_gamma_proton_HERA_uncer_mb_MVe.py
└── gamma_nucleus/
    ├── dipoleMom.py
    ├── dipoleMomNew.py
    ├── dipoleMom_b_dep.py
    ├── sys_L_gamma_nucleus_EIC_posterior_loop.py
    ├── sys_L_gamma_nucleus_EIC_uncertain_MV.py
    ├── sys_T_gamma_nucleus_EIC_posterior_loop.py
    ├── sys_T_gamma_nucleus_EIC_uncertain_MV.py
    └── ws_interpolator.py
```
