# Giải thích chi tiết notebook brain-stroke-classification.ipynb

Tài liệu này giải thích notebook theo từng cell và theo từng dòng code ở các cell code.

## Quy ước đọc

- Đánh số theo thứ tự cell trong notebook (Cell 1, Cell 2, ...).
- Chỉ các cell code có giải thích từng dòng.
- Các cell markdown chỉ mô tả mục tiêu và luồng xử lý.

## Cell 3 - Import thư viện và cấu hình

```python
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
  ConfusionMatrixDisplay,
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DATA_PATH = Path("brain_stroke.csv")
RESULTS_DIR = Path("results")
IMAGES_DIR = Path("img")

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

print(f"Data file exists: {DATA_PATH.exists()}")
```

Giải thích từng dòng:

- `import random`: dùng random chuẩn Python.
- `import sys`: lấy thông tin phiên bản Python ở cuối notebook.
- `import warnings`: quản lý cảnh báo runtime.
- `from pathlib import Path`: thao tác đường dẫn file theo kiểu object.
- `import numpy as np`: xử lý số học/mảng.
- `import pandas as pd`: xử lý dữ liệu bảng.
- `import matplotlib.pyplot as plt`: vẽ chart cơ bản.
- `import seaborn as sns`: vẽ chart thống kê đẹp hơn.
- `import sklearn`: dùng để in version sklearn cho reproducibility.
- `from sklearn.compose import ColumnTransformer`: tiền xử lý theo nhóm cột.
- `from sklearn.impute import SimpleImputer`: điền giá trị thiếu.
- `from sklearn.metrics ...`: nhập toàn bộ metric đánh giá.
- `from sklearn.model_selection ...`: chia train/test và cross-validation.
- `from sklearn.pipeline import Pipeline`: gom các bước preprocessing/model.
- `from sklearn.preprocessing import OneHotEncoder`: mã hóa one-hot cho cột category.
- `from sklearn.tree ...`: model cây + xuất luật + vẽ cây.
- `warnings.filterwarnings("ignore")`: ẩn warning để output gọn.
- `plt.style.use(...)`: set style matplotlib.
- `sns.set_theme(...)`: set style seaborn.
- `RANDOM_STATE = 42`: seed cố định.
- `DATA_PATH = Path("brain_stroke.csv")`: đường dẫn file dữ liệu.
- `RESULTS_DIR = Path("results")`: thư mục xuất bảng kết quả.
- `IMAGES_DIR = Path("img")`: thư mục xuất ảnh.
- `np.random.seed(...)`: cố định random của numpy.
- `random.seed(...)`: cố định random chuẩn Python.
- `RESULTS_DIR.mkdir(exist_ok=True)`: tạo `results/` nếu chưa có.
- `IMAGES_DIR.mkdir(exist_ok=True)`: tạo `img/` nếu chưa có.
- `print(...)`: in trạng thái tồn tại của file CSV.

## Cell 5 - Đọc dữ liệu

```python
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
display(df.head())
```

- `df = pd.read_csv(DATA_PATH)`: nạp CSV vào DataFrame `df`.
- `print("Dataset shape:", df.shape)`: in số dòng, số cột.
- `print("Columns:", list(df.columns))`: in danh sách tên cột.
- `print("\nFirst 5 rows:")`: in tiêu đề output.
- `display(df.head())`: hiển thị 5 dòng đầu.

## Cell 6 - Tổng quan kiểu dữ liệu

```python
print("Data type information:")
df.info()

print("\nDescriptive statistics for numeric columns:")
display(df.describe().T)
```

- `print(...)`: tiêu đề section.
- `df.info()`: số non-null, dtype từng cột, bộ nhớ.
- `print(...)`: tiêu đề thống kê mô tả.
- `display(df.describe().T)`: thống kê cột số, transpose để dễ đọc.

## Cell 8 - Kiểm tra chất lượng dữ liệu

```python
missing_summary = df.isna().sum().sort_values(ascending=False)
duplicate_count = df.duplicated().sum()
class_distribution = df["stroke"].value_counts().sort_index()
class_ratio = (df["stroke"].value_counts(normalize=True).sort_index() * 100).round(2)

print("Missing values by column:")
display(missing_summary.to_frame(name="missing_count"))

print(f"Duplicate rows: {duplicate_count}")

print("\nTarget distribution for `stroke`:")
summary_df = pd.DataFrame({
  "count": class_distribution,
  "ratio_percent": class_ratio,
})
display(summary_df)
```

- `missing_summary = ...`: đếm NA theo cột và sort giảm dần.
- `duplicate_count = ...`: đếm dòng trùng hoàn toàn.
- `class_distribution = ...`: đếm số mẫu mỗi lớp `stroke`.
- `class_ratio = ...`: tỷ lệ phần trăm mỗi lớp.
- `print("Missing values...")`: tiêu đề output NA.
- `display(...)`: hiện bảng NA.
- `print(f"Duplicate rows...")`: in số dòng trùng.
- `print("Target distribution...")`: tiêu đề phân bố target.
- `summary_df = pd.DataFrame(...)`: ghép count + ratio vào bảng.
- `display(summary_df)`: hiển thị bảng phân bố target.

## Cell 10 - Vẽ biểu đồ EDA

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.countplot(data=df, x="stroke", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("Target distribution: stroke")
axes[0, 0].set_xlabel("Stroke")
axes[0, 0].set_ylabel("Sample count")

sns.histplot(data=df, x="age", hue="stroke", bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Age distribution by stroke")

sns.histplot(data=df, x="avg_glucose_level", hue="stroke", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Average glucose by stroke")

sns.histplot(data=df, x="bmi", hue="stroke", bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("BMI distribution by stroke")

plt.tight_layout()
plt.show()
```

- `fig, axes = ...`: tạo canvas 2x2.
- `sns.countplot(...)`: biểu đồ số lượng lớp `stroke`.
- `set_title/xlabel/ylabel`: đặt tiêu đề và nhãn trục plot 1.
- `sns.histplot(... age ...)`: histogram tuổi theo lớp stroke.
- `set_title(...)`: tiêu đề plot 2.
- `sns.histplot(... avg_glucose_level ...)`: histogram glucose theo lớp.
- `set_title(...)`: tiêu đề plot 3.
- `sns.histplot(... bmi ...)`: histogram BMI theo lớp.
- `set_title(...)`: tiêu đề plot 4.
- `plt.tight_layout()`: tránh chồng nhãn.
- `plt.show()`: hiển thị ảnh.

## Cell 12 - Tiền xử lý dữ liệu

```python
df_model = df.copy()

X_raw = df_model.drop(columns=["stroke"])
y = df_model["stroke"]

categorical_columns = X_raw.select_dtypes(include="object").columns.tolist()
numeric_columns = [col for col in X_raw.columns if col not in categorical_columns]
print("Categorical columns to encode:", categorical_columns)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
  X_raw,
  y,
  test_size=0.2,
  random_state=RANDOM_STATE,
  stratify=y,
)

numeric_transformer = Pipeline(
  steps=[
    ("imputer", SimpleImputer(strategy="mean")),
  ]
)

categorical_transformer = Pipeline(
  steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
  ]
)

preprocessor = ColumnTransformer(
  transformers=[
    ("num", numeric_transformer, numeric_columns),
    ("cat", categorical_transformer, categorical_columns),
  ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

feature_names = preprocessor.get_feature_names_out()
X_train = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
X_test = pd.DataFrame(X_test, columns=feature_names, index=X_test_raw.index)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("Stroke rate in train:", round(y_train.mean() * 100, 2), "%")
print("Stroke rate in test :", round(y_test.mean() * 100, 2), "%")
```

- `df_model = df.copy()`: copy dữ liệu để an toàn thao tác.
- `X_raw = ...drop("stroke")`: tách biến đầu vào.
- `y = ...["stroke"]`: tách biến mục tiêu.
- `categorical_columns = ...`: lấy danh sách cột kiểu object.
- `numeric_columns = ...`: lấy các cột còn lại là cột số.
- `print(...)`: in cột categorical.
- `train_test_split(...)`: chia train/test với stratify để giữ tỷ lệ lớp.
- `numeric_transformer = Pipeline(...)`: pipeline cho cột số (mean imputation).
- `categorical_transformer = Pipeline(...)`: pipeline cho cột category (impute + one-hot).
- `preprocessor = ColumnTransformer(...)`: ghép 2 pipeline theo nhóm cột.
- `X_train = preprocessor.fit_transform(...)`: fit + transform train.
- `X_test = preprocessor.transform(...)`: transform test bằng rule của train.
- `feature_names = ...get_feature_names_out()`: lấy tên cột sau encode.
- `X_train = pd.DataFrame(...)`: chuyển ma trận train thành DataFrame.
- `X_test = pd.DataFrame(...)`: chuyển ma trận test thành DataFrame.
- `print(...)`: in shape và tỷ lệ stroke train/test.

## Cell 14 - Hàm đánh giá dùng chung

### 14.1 `evaluate_tree_model(...)`

- `model.fit(X_train, y_train)`: huấn luyện model.
- `y_pred = model.predict(X_test)`: dự đoán nhãn.
- `y_prob = model.predict_proba(X_test)[:, 1]`: xác suất lớp 1.
- `cm = confusion_matrix(...)`: tạo confusion matrix.
- `tn, fp, fn, tp = cm.ravel()`: tách 4 ô ma trận.
- Khối `result = {...}`: gom toàn bộ metrics cần báo cáo.
- `return result`: trả dict kết quả.

### 14.2 `show_confusion_matrix(...)`

- Tạo `ConfusionMatrixDisplay`.
- `disp.plot(...)`: vẽ matrix.
- `plt.grid(False)`: tắt grid.
- Nếu có `save_path`: `savefig` ra file.
- `plt.show()`: hiển thị.

### 14.3 `summarize_overfitting(...)`

- Tính `gap = train_acc - test_acc`.
- Nếu `gap >= 0.10`: overfitting rõ.
- Nếu `gap >= 0.05`: overfitting nhẹ.
- Ngược lại: chưa đáng lo.

### 14.4 `result_to_frame(...)`

- Tạo DataFrame 2 cột `Metric` và `Value`.
- Chuẩn hóa output để `display(...)` đẹp và đồng nhất.

## Cell 16 - Huấn luyện baseline

```python
baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
baseline_result = evaluate_tree_model(...)

print("=== BASELINE MODEL RESULTS ===")
display(result_to_frame(baseline_result))

print("Classification report:")
print(classification_report(y_test, baseline_result["y_pred"], zero_division=0))
```

- `DecisionTreeClassifier(...)`: khởi tạo baseline mặc định.
- `evaluate_tree_model(...)`: train + tính metrics baseline.
- `print/display(...)`: in bảng metrics.
- `classification_report(...)`: in precision/recall/F1 từng lớp.

## Cell 17 - Confusion matrix và hình cây baseline

- Gọi `show_confusion_matrix(...)` cho baseline.
- `plt.figure(...)`: tạo figure lớn để nhìn cây.
- `plot_tree(...)`: vẽ tree (giới hạn `max_depth=3`).
- `savefig(...)`: lưu `img/baseline_tree_top3.png`.
- `plt.show()`: hiển thị.

## Cell 19 - Phân tích cây baseline

- `baseline_tree = baseline_result["model"]`: lấy object cây.
- `root_feature_index/root_threshold`: lấy thuộc tính tách ở node gốc.
- `feature_importance = ...head(10)`: top 10 đặc trưng quan trọng.
- `print(...)`: in depth, leaves, root split, FNR, overfitting summary.
- `baseline_rules = export_text(...)`: xuất luật cây dạng text.
- `write_text(...)`: lưu `results/baseline_rules.txt`.
- `to_csv(...)`: lưu `results/feature_importance.csv`.

## Cell 20 - Nhận xét report-ready

- `top3 = feature_importance.head(3)...`: lấy top 3 feature.
- Các lệnh `print(...)`: in nhận xét mẫu để chèn báo cáo.

## Cell 22 - Chạy các mô hình cải tiến

### Khai báo model_candidates

- `Baseline`: model mặc định.
- `Improvement 1`: thêm `class_weight="balanced"`.
- `Improvement 2`: pruning bằng `max_depth=5`, `min_samples_leaf=10`.
- `Improvement 3`: `criterion="entropy"`.
- `Final Selected Model`: kết hợp cả class weight + entropy + pruning.

### Vòng lặp train/evaluate

- `all_results = {}`: dict chứa kết quả mọi model.
- `for model_name, model in model_candidates.items(): ...`: train từng model.
- Lưu output của từng model vào `all_results`.

### Tạo bảng so sánh

- `comparison_rows = []`: danh sách record bảng.
- Mỗi record chứa Accuracy, Recall, F1, ROC-AUC, FNR, depth, leaves.
- `comparison_df = pd.DataFrame(...).sort_values(...)`: ưu tiên sort theo F1, Recall, ROC-AUC.
- `display(comparison_df.round(4))`: hiển thị bảng so sánh.

## Cell 23 - Vẽ chart so sánh model

- Tạo 4 barplot cho Accuracy, Recall, F1, ROC-AUC.
- Mục tiêu: nhìn nhanh trade-off giữa các model.

## Cell 24 - Chốt model được chọn

- Lấy model tốt nhất theo từng metric (`best_accuracy_model`, `best_f1_model`, `best_recall_model`).
- `selected_model_name = comparison_df.iloc[0]["Model"]`: model đứng đầu theo tiêu chí sort.
- In kết luận + guideline chọn model cho bài toán mất cân bằng.

## Cell 26 - Summary cuối, export artifact, cross-validation

### 26.1 Lưu bảng tổng hợp

- Copy `comparison_df` thành `final_summary_df`.
- Thêm cột `Selected` đánh dấu model được chọn.
- Lưu `results/metrics_summary.csv`.

### 26.2 Lưu classification report

- Tạo report cho baseline và model chọn.
- Ghi vào `results/classification_report.txt`.

### 26.3 Lưu confusion matrix

- Lưu baseline matrix thành `img/confusion_baseline.png`.
- Lưu final matrix thành `img/confusion_best_model.png`.

### 26.4 Export rules model cuối

- `export_text(..., max_depth=6)` cho model được chọn.
- Lưu vào `results/final_model_rules.txt`.

### 26.5 Cross-validation leakage-safe

- Tạo `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Dựng lại `preprocessor_cv` tương tự preprocessing chính.
- Dựng `cv_pipeline = [preprocessor -> model]`.
- `cross_validate(...)` với scoring Accuracy, Recall, F1.
- In mean/std các metric CV.

### 26.6 In thông tin tái lập

- In Python version, pandas version, sklearn version, random_state.
- In thông báo đã lưu artifact.

## Cell 28 - Trực quan model cuối

- Lấy `improved_tree_result = all_results[selected_model_name]`.
- Vẽ confusion matrix của model chọn.
- Vẽ top 3 tầng cây model chọn.
- Lưu ảnh `img/improved_tree_top3.png`.

## Các file đầu ra quan trọng

Trong `results/`:

- `metrics_summary.csv`
- `classification_report.txt`
- `baseline_rules.txt`
- `final_model_rules.txt`
- `feature_importance.csv`

Trong `img/`:

- `baseline_tree_top3.png`
- `improved_tree_top3.png`
- `confusion_baseline.png`
- `confusion_best_model.png`

## Gợi ý dùng khi thuyết trình

- Bám theo thứ tự: Data quality -> Baseline -> Tree interpretation -> Improvements -> Final selection.
- Nhấn mạnh lý do ưu tiên Recall/F1/FNR cho bài toán y tế mất cân bằng lớp.
