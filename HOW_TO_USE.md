# How to use the SNN vs CNN research tool

A plain-language guide for the researcher. The data files in `results/` contain
no instructions — this file holds them, so the data files can be pasted straight
into an LLM.

## 1. Run the tool

```bash
python3 snn_cnn_lab.py            # opens the graphical app
```

Pick ONE of the three study buttons and press it — nothing else to configure.
Each study tests both networks (CNN and SNN) across every environment (input
noise, weight precision, and SNN time-steps).

| Study | What it is | Roughly |
|-------|------------|---------|
| Quick Demo | Small subset, a few environments — check it runs | a few minutes |
| Normal Study | Every environment, balanced training — solid report data | ~10–40 min |
| Full Paper Study | Full 60,000-image dataset, every environment, repeated seeds — the data to report | run overnight |

Command-line equivalents (optional):

```bash
python3 snn_cnn_lab.py --study quick    # or normal / full
python3 snn_cnn_lab.py --clean          # delete everything in results/
python3 snn_cnn_lab.py --summary-only   # rebuild reports/plots from existing data
```

The app also has buttons to open the results folder, rebuild the reports, and
delete all results.

## 2. Where the outputs go (the `results/` folder)

- **`Master_Report.md`** — the file to give your LLM. Self-contained: methodology,
  all findings and analysis, aggregated breakdowns, confusion matrices, and the
  full per-run results table. Kept deliberately token-light.
- `REPORT.md` — a heavier variant that also embeds every raw per-run log; use
  only if you specifically want every raw row.
- `master_ledger.csv` and the other CSV logs — the raw data.
- `plots/` — all figures (described inside the reports).
- `METHODOLOGY.md` — definitions of every metric.

## 3. Give it to an LLM

Attach **`results/Master_Report.md`** and use a prompt like:

> Using the attached Master_Report.md from my CNN-vs-SNN MNIST experiments, help
> me write up the research (Introduction, Methodology, Results, Analysis,
> Conclusion, Evaluation). Cite specific figures, tables and numbers from the
> file; explain the accuracy/energy trade-off and the precise conditions under
> which the SNN is or is not more efficient or more robust; and keep every claim
> to what the data supports.

Tip: for the most credible numbers (with error bars), run the **Full Paper Study**
overnight before generating the report you rely on.
