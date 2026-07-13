# SNN vs CNN — Energy/Accuracy Laboratory

A single, self-contained PyTorch application that trains and evaluates
**topologically identical** Convolutional Neural Networks (CNNs) and Spiking
Neural Networks (SNNs) on MNIST, and measures the trade-off between
**classification accuracy** and **estimated inference energy** under a range of
simulated power-constrained conditions.

> **Research question**
> To what extent does the energy efficiency and classification accuracy of a
> biologically-inspired Spiking Neural Network (SNN) compare to a traditional
> Convolutional Neural Network (CNN) when processing MNIST image datasets under
> simulated power-constrained environments?

Everything — the models, the training loop, the SNN dynamics, the energy model,
the logging, the plots and the reports — lives in one file:
[`snn_cnn_lab.py`](snn_cnn_lab.py). There is **no external SNN library** (no
`snntorch`); the spiking network is built from scratch in pure PyTorch so every
mechanism is transparent and reproducible.

## What it measures

Both architectures share an identical layer topology, so their dense
multiply–accumulate (MAC) count is identical. The tool sweeps three proxies for a
"power-constrained environment":

1. **Input noise** — additive Gaussian sensor/thermal noise at inference.
2. **Weight quantization** — low-precision (2/4/8/32-bit) quantization-aware arithmetic.
3. **SNN time-steps (T)** — the temporal budget of the spiking network.

For every configuration it logs 40+ metrics per run, including accuracy, measured
firing rates, spike operations (SOPs), and an analytic energy estimate.

### Energy model

Energy is estimated analytically from operation counts using the widely-cited
45 nm figures of Horowitz (ISSCC 2014):

| Operation | Energy (32-bit) |
|-----------|-----------------|
| Float multiply (`E_MULT`) | 3.7 pJ |
| Float add (`E_ADD`)       | 0.9 pJ |
| MAC = mult + add (`E_MAC`) | 4.6 pJ — used by the dense CNN |
| AC = add only (`E_AC`)     | 0.9 pJ — used by the event-driven SNN |

- **CNN energy** = dense MACs × `E_MAC(bits)`
- **SNN energy** = sparse SOPs × `E_AC(bits)`

Using the standard SNN identity (Rathi & Roy; Lemaire et al.), for any layer *l*:

```
SOPs_l = firing_rate_l × T × MACs_l
```

`firing_rate_l` is **measured empirically** on the test set; `MACs_l` is computed
analytically. Multiplier energy is assumed to scale ~quadratically and adder
energy ~linearly with bit-width. BatchNorm is folded into the preceding
conv/linear at inference, so it adds no extra inference operations. See
[`results/METHODOLOGY.md`](results/METHODOLOGY.md) (generated on first run) for
the full, formal write-up.

## Installation

Requires **Python 3.9+**.

```bash
git clone <your-repo-url>
cd EE_Project
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

The optional GUI uses Tkinter, which ships with the standard CPython installers
on macOS and Windows. On Debian/Ubuntu: `sudo apt-get install python3-tk`. The
command-line mode does not need Tkinter.

### Getting the MNIST data

The tool reads MNIST directly from the raw IDX files and expects them in a
`MNIST Dataset/` folder next to the script:

```
MNIST Dataset/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

The dataset is not committed to this repository (it is large and freely
available). Download the four files from any MNIST mirror (e.g.
<https://ossci-datasets.s3.amazonaws.com/mnist/>) and place them as shown above,
or point the tool at another location with `--data-dir /path/to/mnist`.

## Usage

### Graphical app

```bash
python3 snn_cnn_lab.py
```

Pick one of three preset studies and press its button — no other configuration
needed:

| Study | Coverage | Runtime |
|-------|----------|---------|
| **Quick Demo** | Small subset, a few environments — confirms it runs | a few minutes |
| **Normal Study** | Every environment, balanced training on a 15k subset | ~10–40 min |
| **Full Paper Study** | Full 60,000-image dataset, every environment, repeated seeds (error bars) | run overnight |

### Command line

```bash
python3 snn_cnn_lab.py --study quick        # or: normal / full
python3 snn_cnn_lab.py --headless \
    --models CNN,SNN --noises 0,0.25 --quants 32,8 --timesteps 25,50
python3 snn_cnn_lab.py --summary-only       # rebuild reports/plots from existing data
python3 snn_cnn_lab.py --clean              # delete everything in results/
```

## Outputs

Everything lands in `results/` (git-ignored — regenerate any time):

| File | What it is |
|------|------------|
| `Master_Report.md` | Curated, self-contained report: methodology, all findings, aggregated breakdowns, confusion matrices, and the full per-run table. |
| `REPORT.md` | Heavier variant that also embeds every raw per-run log. |
| `master_ledger.csv` | One row per run — the high-level overview (40+ metrics). |
| `epoch_log.csv` / `layer_log.csv` / `class_log.csv` / `confusion_log.csv` | Per-epoch, per-layer, per-class and confusion-matrix detail. |
| `executive_summary.{csv,txt}` | CNN-vs-SNN comparison averaged over repeats. |
| `METHODOLOGY.md` | Plain-language definition of every metric and pipeline step. |
| `plots/*.png` | Trade-off curves, learning curves, heatmaps, confusion matrices. |

See [`HOW_TO_USE.md`](HOW_TO_USE.md) for a step-by-step walkthrough, including how
to hand the generated report to an LLM to help draft a write-up.

## Reproducibility

- Single self-contained script — no hidden state, no external SNN framework.
- Fixed random seeds per configuration; the Full Paper Study repeats seeds for error bars.
- The exact software environment (Python/PyTorch/NumPy/pandas/matplotlib versions)
  is captured into the generated methodology report on every run.

## References

- M. Horowitz, "Computing's Energy Problem (and what we can do about it)," *ISSCC*, 2014.
- N. Rathi and K. Roy, "DIET-SNN," *IEEE TNNLS*, 2021.
- E. Lemaire et al., "An Analytical Estimation of Spiking Neural Networks Energy Efficiency," 2022.

## License

Released under the [MIT License](LICENSE).
