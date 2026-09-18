# Integrated Bayesian Multimodal Longitudinal Physiological Analysis — Simulation Code

Supporting data and code for the manuscript:

> *Integrated Bayesian Multimodal Longitudinal Physiological Analysis for Human Spaceflight Research*

## Contents

| Path | Description |
|------|-------------|
| `simulate_and_evaluate.py` | End-to-end synthetic mission simulation, model fit, baselines, metrics, and figure generation |
| `requirements.txt` | Python dependencies |
| `results/metrics.json` | Numeric evaluation metrics (seed 42) |
| `results/summary.md` | Human-readable results summary |
| `figures/` | Figures 1–4 used in the manuscript |

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulate_and_evaluate.py
```

Fixed random seed: **42**.

## Important disclaimer

All results in this repository are from a **controlled synthetic simulation** (8 crew × 90 mission days).  
**No NASA operational crew data, Artemis flight telemetry, or ISS astronaut health records were used.**

## Citation

If you use this code or metrics, please cite the accompanying manuscript and this repository.

## License

MIT License (see `LICENSE`).

## Author

Arpan Bom
