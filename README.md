# Decision Tree Stroke Classification

This project uses `DecisionTreeClassifier` to predict `stroke` outcomes from the `brain_stroke.csv` dataset. The main workflow is provided in `brain-stroke-classification.ipynb`, which is organized in a report-friendly sequence: data loading, exploratory analysis, preprocessing, baseline model training, tree interpretation, and model improvement.

## Project Objectives

- Build a baseline Decision Tree model for stroke prediction
- Analyze the structure and behavior of the decision tree
- Compare the baseline model with improved versions such as `class_weight`, `pruning`, `entropy`, and a combined final model
- Support lab work, coursework, and report writing with a reproducible workflow

## Project Structure

- `brain-stroke-classification.ipynb`: main notebook containing the full analysis and modeling process
- `brain_stroke.csv`: input dataset
- `requirements.txt`: required Python packages for the project

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Environment Setup

It is recommended to use a virtual environment to avoid dependency conflicts.

### 1. Clone or download the project, then open a terminal in the project folder

```bash
cd Decision-Tree-Stroke
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
```

On Windows:

```bash
py -m venv .venv
```

### 3. Activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows Command Prompt:

```bash
.venv\Scripts\activate
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

After activation, your terminal prompt will usually show `(.venv)`.

### 4. Upgrade `pip`

On macOS or Linux:

```bash
python -m pip install --upgrade pip
```

On Windows:

```bash
py -m pip install --upgrade pip
```

### 5. Install all required packages

```bash
pip install -r requirements.txt
```

## How to Run the Project

After the environment is ready, start Jupyter Notebook.

On macOS or Linux:

```bash
source .venv/bin/activate
jupyter notebook
```

On Windows:

```bash
.venv\Scripts\activate
jupyter notebook
```

Then:

1. Jupyter Notebook will open in your browser
2. Open `brain-stroke-classification.ipynb`
3. Run each cell with `Shift + Enter`
4. If you want to rerun everything from the beginning, use `Kernel -> Restart & Run All`

## Notebook Workflow

- Load the dataset from `brain_stroke.csv`
- Inspect dataset size, columns, data types, missing values, and duplicates
- Visualize the distributions of `stroke`, `age`, `avg_glucose_level`, and `bmi`
- Preprocess the data by:
  - splitting data first with `stratify` to avoid leakage
  - using `ColumnTransformer` + `Pipeline`
  - imputing numeric columns with mean
  - imputing categorical columns with most frequent + one-hot encoding
- Train a baseline `DecisionTreeClassifier`
- Evaluate models using Accuracy, Error Rate, Precision, Recall, F1-Score, ROC-AUC, and False Negative Rate (FNR)
- Visualize the decision tree and inspect feature importance
- Compare multiple improved Decision Tree configurations
- Select a final model based on imbalanced-classification priorities (Recall/F1)
- Run 5-fold `StratifiedKFold` cross-validation for final-model stability

## Generated Outputs

After running the notebook, artifacts are exported automatically:

- `results/metrics_summary.csv`
- `results/classification_report.txt`
- `results/baseline_rules.txt`
- `results/final_model_rules.txt`
- `results/feature_importance.csv`
- `img/baseline_tree_top3.png`
- `img/improved_tree_top3.png`
- `img/confusion_baseline.png`
- `img/confusion_best_model.png`

These files provide the outputs used in the report and presentation.

## Troubleshooting

### `jupyter` command not found

Make sure the virtual environment is activated before running Jupyter:

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Package installation issues with very new Python versions

If `scikit-learn` or another package fails to install, using Python `3.11` or `3.12` is recommended instead of a very new release.

### Exit the virtual environment

On macOS or Linux:

```bash
deactivate
```

On Windows, the same command usually works:

```bash
deactivate
```

## Usage Notes

- Run the cells one by one if you want to follow the workflow step by step
- If you modify the code and want to verify everything again, use `Restart & Run All`
- For imbalanced classification tasks like stroke prediction, do not rely only on Accuracy; prioritize Recall/F1 and also inspect FNR

## Notes

This README documents the setup and usage of the project for study, lab, and report purposes.
