# Decision Tree Stroke Classification

Submission-ready lab project for Decision Tree modeling and improvement on the brain stroke dataset.

## What this notebook covers

- Dataset loading, quality checks, and preprocessing
- Train/test split with stratification
- Baseline `DecisionTreeClassifier`
- Tree analysis and visualization
- Three improvements:
  - `class_weight="balanced"`
  - pruning (`max_depth`, `min_samples_leaf`)
  - `criterion="entropy"`
- Final selected model: combined settings (`class_weight + entropy + pruning`)
- Baseline vs improved comparison table
- Additional robustness check with `StratifiedKFold` cross-validation

## Project structure

- `brain-stroke-classification.ipynb`: main workflow
- `brain_stroke.csv`: dataset
- `requirements.txt`: reproducible dependencies
- `results/`: exported tables and text outputs
- `img/`: exported figures for report/video

## Setup

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Command Prompt)

```bash
py -m venv .venv
.venv\Scripts\activate.bat
py -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook
```

Open `brain-stroke-classification.ipynb` and run all cells (`Restart & Run All`).

## Generated artifacts

Main outputs are saved automatically after execution:

- `results/metrics_summary.csv`
- `results/final_model_cv_summary.csv`
- `results/classification_reports.txt`
- `results/baseline_rules.txt`
- `results/final_model_rules.txt`
- `results/feature_importance.csv`
- `results/environment_info.txt`
- `img/baseline_tree_top3.png`
- `img/final_tree_top3.png`
- `img/confusion_baseline.png`
- `img/confusion_final.png`
- `img/comparison_metrics.png`

## Notes for report/video

- Final selected model configuration is documented in the notebook and comparison table.
- Use `results/metrics_summary.csv` as the main baseline-vs-improved experiment summary.
- Use `results/final_model_cv_summary.csv` for cross-validation stability evidence.
