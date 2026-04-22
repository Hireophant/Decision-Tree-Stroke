# Giải Thích Notebook `brain-stroke-classification.ipynb`

File này giải thích từng code cell trong notebook bằng tiếng Việt.  
Mục tiêu là để bạn biết mỗi cell làm gì và từng dòng code có ý nghĩa gì.

## Cell 2: Import thư viện và cấu hình ban đầu

```python
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DATA_PATH = Path("brain_stroke.csv")

print(f"Data file exists: {DATA_PATH.exists()}")
```

Giải thích từng dòng:

- `import warnings`
  Dùng module `warnings` để quản lý hoặc ẩn các cảnh báo khi chạy code.

- `from pathlib import Path`
  Import `Path` để thao tác với đường dẫn file gọn hơn so với string thường.

- `import numpy as np`
  Import thư viện `numpy` và đặt bí danh là `np`. Thư viện này hỗ trợ tính toán mảng số.

- `import pandas as pd`
  Import `pandas` với bí danh `pd`. Đây là thư viện chính để đọc và xử lý dữ liệu dạng bảng.

- `import matplotlib.pyplot as plt`
  Import phần vẽ biểu đồ của `matplotlib`, đặt tên ngắn là `plt`.

- `import seaborn as sns`
  Import `seaborn`, thư viện vẽ biểu đồ thống kê đẹp và tiện hơn trên nền `matplotlib`.

- `from sklearn.metrics import (...)`
  Import các hàm đánh giá mô hình từ `scikit-learn`.

- `ConfusionMatrixDisplay`
  Dùng để vẽ confusion matrix.

- `accuracy_score`
  Tính độ chính xác tổng thể của mô hình.

- `classification_report`
  In ra bảng tóm tắt precision, recall, f1-score cho từng lớp.

- `confusion_matrix`
  Tạo ma trận nhầm lẫn giữa nhãn thật và nhãn dự đoán.

- `f1_score`
  Tính F1-score, cân bằng giữa precision và recall.

- `precision_score`
  Tính precision, tức là dự đoán dương tính thì đúng bao nhiêu phần trăm.

- `recall_score`
  Tính recall, tức là bắt được bao nhiêu mẫu dương tính thật.

- `roc_auc_score`
  Tính điểm ROC-AUC, đo khả năng phân biệt hai lớp của mô hình.

- `from sklearn.model_selection import train_test_split`
  Import hàm chia dữ liệu thành tập train và test.

- `from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree`
  Import mô hình cây quyết định, hàm in luật cây ra text, và hàm vẽ cây.

- `warnings.filterwarnings("ignore")`
  Ẩn bớt warning để output gọn hơn khi chạy notebook.

- `plt.style.use("seaborn-v0_8-whitegrid")`
  Đặt style mặc định cho biểu đồ `matplotlib`.

- `sns.set_theme(style="whitegrid")`
  Đặt theme mặc định cho biểu đồ `seaborn`.

- `RANDOM_STATE = 42`
  Tạo hằng số để cố định random seed, giúp chạy lại nhiều lần vẫn ra kết quả giống nhau.

- `DATA_PATH = Path("brain_stroke.csv")`
  Tạo biến chứa đường dẫn tới file dữ liệu.

- `print(f"Data file exists: {DATA_PATH.exists()}")`
  Kiểm tra file CSV có tồn tại không rồi in kết quả ra màn hình.

## Cell 4: Đọc dữ liệu

```python
df = pd.read_csv(DATA_PATH)

print("Kích thước dữ liệu:", df.shape)
print("Các cột:", list(df.columns))
print("\n5 dòng đầu tiên:")
display(df.head())
```

Giải thích từng dòng:

- `df = pd.read_csv(DATA_PATH)`
  Đọc file CSV vào DataFrame tên là `df`.

- `print("Kích thước dữ liệu:", df.shape)`
  In ra số dòng và số cột của dataset.

- `print("Các cột:", list(df.columns))`
  In danh sách tên các cột để biết dữ liệu gồm những thuộc tính nào.

- `print("\n5 dòng đầu tiên:")`
  In tiêu đề trước khi hiển thị dữ liệu mẫu.

- `display(df.head())`
  Hiển thị 5 dòng đầu tiên của DataFrame trong notebook.

## Cell 5: Xem thông tin tổng quan của dữ liệu

```python
print("Thông tin kiểu dữ liệu:")
df.info()

print("\nThống kê mô tả cho các cột số:")
display(df.describe().T)
```

Giải thích từng dòng:

- `print("Thông tin kiểu dữ liệu:")`
  In dòng tiêu đề để dễ nhìn output.

- `df.info()`
  Hiển thị số lượng non-null, kiểu dữ liệu của từng cột, và bộ nhớ sử dụng.

- `print("\nThống kê mô tả cho các cột số:")`
  In tiêu đề cho phần thống kê mô tả.

- `display(df.describe().T)`
  Tính các thống kê như count, mean, std, min, max cho các cột số rồi xoay bảng bằng `.T` để dễ đọc.

## Cell 7: Kiểm tra chất lượng dữ liệu

```python
missing_summary = df.isna().sum().sort_values(ascending=False)
duplicate_count = df.duplicated().sum()
class_distribution = df["stroke"].value_counts().sort_index()
class_ratio = (df["stroke"].value_counts(normalize=True).sort_index() * 100).round(2)

print("Số lượng giá trị thiếu theo từng cột:")
display(missing_summary.to_frame(name="missing_count"))

print(f"Số dòng bị trùng: {duplicate_count}")

print("\nPhân bố biến mục tiêu `stroke`:")
summary_df = pd.DataFrame({
    "count": class_distribution,
    "ratio_percent": class_ratio,
})
display(summary_df)
```

Giải thích từng dòng:

- `missing_summary = df.isna().sum().sort_values(ascending=False)`
  Kiểm tra giá trị thiếu ở từng cột, cộng tổng số ô bị thiếu, rồi sắp xếp giảm dần.

- `duplicate_count = df.duplicated().sum()`
  Đếm xem có bao nhiêu dòng bị trùng hoàn toàn.

- `class_distribution = df["stroke"].value_counts().sort_index()`
  Đếm số mẫu của từng lớp trong cột `stroke`, rồi sắp xếp theo nhãn lớp.

- `class_ratio = (df["stroke"].value_counts(normalize=True).sort_index() * 100).round(2)`
  Tính tỷ lệ phần trăm của từng lớp trong cột `stroke` và làm tròn 2 chữ số.

- `print("Số lượng giá trị thiếu theo từng cột:")`
  In tiêu đề cho phần missing values.

- `display(missing_summary.to_frame(name="missing_count"))`
  Chuyển `Series` thành DataFrame một cột tên `missing_count` rồi hiển thị.

- `print(f"Số dòng bị trùng: {duplicate_count}")`
  In ra số dòng trùng.

- `print("\nPhân bố biến mục tiêu `stroke`:")`
  In tiêu đề cho phần phân bố của biến đích.

- `summary_df = pd.DataFrame({...})`
  Tạo một bảng mới gồm số lượng mẫu và tỷ lệ phần trăm của từng lớp.

- `"count": class_distribution`
  Cột `count` chứa số lượng mẫu mỗi lớp.

- `"ratio_percent": class_ratio`
  Cột `ratio_percent` chứa tỷ lệ phần trăm của từng lớp.

- `display(summary_df)`
  Hiển thị bảng tổng hợp vừa tạo.

## Cell 9: Vẽ biểu đồ khám phá dữ liệu

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# A quick look at the main distributions in the dataset.
sns.countplot(data=df, x="stroke", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("Stroke class distribution")
axes[0, 0].set_xlabel("Stroke")
axes[0, 0].set_ylabel("Number of samples")

sns.histplot(data=df, x="age", hue="stroke", bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Age distribution by stroke")

sns.histplot(data=df, x="avg_glucose_level", hue="stroke", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Glucose distribution by stroke")

sns.histplot(data=df, x="bmi", hue="stroke", bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("BMI distribution by stroke")

plt.tight_layout()
plt.show()
```

Giải thích từng dòng:

- `fig, axes = plt.subplots(2, 2, figsize=(16, 10))`
  Tạo một khung hình gồm 4 ô biểu đồ theo dạng 2 hàng 2 cột.

- `# A quick look at the main distributions in the dataset.`
  Comment cho biết phần dưới đang vẽ các phân bố chính trong dữ liệu.

- `sns.countplot(data=df, x="stroke", palette="Set2", ax=axes[0, 0])`
  Vẽ biểu đồ cột đếm số lượng mẫu của từng lớp `stroke`.

- `axes[0, 0].set_title("Stroke class distribution")`
  Đặt tiêu đề cho biểu đồ đầu tiên.

- `axes[0, 0].set_xlabel("Stroke")`
  Đặt nhãn trục X.

- `axes[0, 0].set_ylabel("Number of samples")`
  Đặt nhãn trục Y.

- `sns.histplot(data=df, x="age", hue="stroke", bins=30, kde=True, ax=axes[0, 1])`
  Vẽ histogram của `age`, tô màu theo lớp `stroke`, đồng thời thêm đường KDE.

- `axes[0, 1].set_title("Age distribution by stroke")`
  Đặt tiêu đề cho biểu đồ tuổi.

- `sns.histplot(data=df, x="avg_glucose_level", hue="stroke", bins=30, kde=True, ax=axes[1, 0])`
  Vẽ phân bố `avg_glucose_level` theo lớp `stroke`.

- `axes[1, 0].set_title("Glucose distribution by stroke")`
  Đặt tiêu đề cho biểu đồ glucose.

- `sns.histplot(data=df, x="bmi", hue="stroke", bins=30, kde=True, ax=axes[1, 1])`
  Vẽ phân bố `bmi` theo lớp `stroke`.

- `axes[1, 1].set_title("BMI distribution by stroke")`
  Đặt tiêu đề cho biểu đồ BMI.

- `plt.tight_layout()`
  Tự căn chỉnh khoảng cách giữa các biểu đồ để không bị chồng chữ.

- `plt.show()`
  Hiển thị toàn bộ hình.

## Cell 11: Tiền xử lý dữ liệu

```python
df_model = df.copy()

# Fill missing BMI values with the mean.
if df_model["bmi"].isna().sum() > 0:
    df_model["bmi"] = df_model["bmi"].fillna(df_model["bmi"].mean())

# Turn categorical columns into numeric features.
categorical_columns = df_model.select_dtypes(include="object").columns.tolist()
print("Categorical columns to encode:", categorical_columns)

df_encoded = pd.get_dummies(df_model, columns=categorical_columns, drop_first=True)

# Separate features and target.
X = df_encoded.drop(columns=["stroke"])
y = df_encoded["stroke"]

# Keep the class ratio similar in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("Stroke rate in train:", round(y_train.mean() * 100, 2), "%")
print("Stroke rate in test :", round(y_test.mean() * 100, 2), "%")
```

Giải thích từng dòng:

- `df_model = df.copy()`
  Tạo một bản sao của dữ liệu gốc để xử lý, tránh làm thay đổi `df`.

- `# Fill missing BMI values with the mean.`
  Comment cho biết phần dưới sẽ xử lý missing value của BMI.

- `if df_model["bmi"].isna().sum() > 0:`
  Kiểm tra xem cột `bmi` có giá trị thiếu hay không.

- `df_model["bmi"] = df_model["bmi"].fillna(df_model["bmi"].mean())`
  Điền các giá trị thiếu trong `bmi` bằng giá trị trung bình của cột đó.

- `# Turn categorical columns into numeric features.`
  Comment cho biết phần tiếp theo là mã hóa các cột phân loại.

- `categorical_columns = df_model.select_dtypes(include="object").columns.tolist()`
  Lấy danh sách tất cả cột có kiểu dữ liệu chuỗi để chuẩn bị encode.

- `print("Categorical columns to encode:", categorical_columns)`
  In ra tên các cột phân loại.

- `df_encoded = pd.get_dummies(df_model, columns=categorical_columns, drop_first=True)`
  One-hot encode các cột phân loại, mỗi giá trị sẽ thành cột nhị phân.
  `drop_first=True` giúp bỏ bớt một cột ở mỗi nhóm để tránh dư thừa thông tin.

- `# Separate features and target.`
  Comment cho biết bắt đầu tách đầu vào và đầu ra.

- `X = df_encoded.drop(columns=["stroke"])`
  Tạo biến đầu vào `X` bằng cách bỏ cột mục tiêu `stroke`.

- `y = df_encoded["stroke"]`
  Gán biến mục tiêu `y` là cột `stroke`.

- `# Keep the class ratio similar in both train and test sets.`
  Comment cho biết sẽ chia dữ liệu có giữ tỷ lệ lớp.

- `X_train, X_test, y_train, y_test = train_test_split(...)`
  Chia dữ liệu thành tập train và test.

- `X,`
  Dữ liệu đầu vào để chia.

- `y,`
  Nhãn mục tiêu tương ứng.

- `test_size=0.2,`
  Dành 20% dữ liệu cho tập test.

- `random_state=RANDOM_STATE,`
  Cố định việc random để chia dữ liệu ổn định qua các lần chạy.

- `stratify=y,`
  Giữ tỷ lệ lớp `stroke` gần giống nhau ở tập train và test.

- `print("X_train shape:", X_train.shape)`
  In kích thước tập train.

- `print("X_test shape :", X_test.shape)`
  In kích thước tập test.

- `print("Stroke rate in train:", round(y_train.mean() * 100, 2), "%")`
  Tính tỷ lệ lớp `stroke = 1` trong train và in ra phần trăm.

- `print("Stroke rate in test :", round(y_test.mean() * 100, 2), "%")`
  Tính tỷ lệ lớp `stroke = 1` trong test và in ra phần trăm.

## Cell 13: Các hàm hỗ trợ

```python
def evaluate_tree_model(model_name, model, X_train, X_test, y_train, y_test):
    # Train model.
    model.fit(X_train, y_train)

    # Predict labels and probabilities on the test set.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Store the main metrics together for easier comparison.
    result = {
        "model_name": model_name,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "accuracy": accuracy_score(y_test, y_pred),
        "error_rate": 1 - accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
    }
    return result


def show_confusion_matrix(cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.grid(False)
    plt.show()


def summarize_overfitting(train_acc, test_acc):
    gap = train_acc - test_acc
    if gap >= 0.10:
        return f"Train accuracy is higher than test accuracy by {gap:.4f}, which suggests clear overfitting."
    if gap >= 0.05:
        return f"The train/test gap is {gap:.4f}, so there may be mild overfitting."
    return f"The train/test gap is {gap:.4f}, so overfitting does not look too severe."


def result_to_frame(result):
    return pd.DataFrame(
        {
            "Metric": ["Train Accuracy", "Accuracy", "Error Rate", "Precision", "Recall", "F1-Score", "ROC-AUC", "Tree Depth", "Number of Leaves"],
            "Value": [
                result["train_accuracy"],
                result["accuracy"],
                result["error_rate"],
                result["precision"],
                result["recall"],
                result["f1_score"],
                result["roc_auc"],
                result["depth"],
                result["n_leaves"],
            ],
        }
    )
```

Giải thích từng dòng:

- `def evaluate_tree_model(model_name, model, X_train, X_test, y_train, y_test):`
  Định nghĩa hàm để huấn luyện một mô hình cây và tính các chỉ số đánh giá.

- `model.fit(X_train, y_train)`
  Huấn luyện mô hình trên tập train.

- `y_pred = model.predict(X_test)`
  Dự đoán nhãn 0 hoặc 1 cho tập test.

- `y_prob = model.predict_proba(X_test)[:, 1]`
  Lấy xác suất thuộc lớp 1 của từng mẫu test.

- `result = { ... }`
  Tạo một dictionary để gom toàn bộ kết quả.

- `"model_name": model_name`
  Lưu tên mô hình.

- `"model": model`
  Lưu chính object mô hình đã train.

- `"y_pred": y_pred`
  Lưu nhãn dự đoán.

- `"y_prob": y_prob`
  Lưu xác suất dự đoán lớp dương tính.

- `"confusion_matrix": confusion_matrix(y_test, y_pred)`
  Tính ma trận nhầm lẫn giữa nhãn thật và nhãn dự đoán.

- `"train_accuracy": accuracy_score(y_train, model.predict(X_train))`
  Tính accuracy trên tập train để xem mô hình học dữ liệu train tốt thế nào.

- `"accuracy": accuracy_score(y_test, y_pred)`
  Tính accuracy trên tập test.

- `"error_rate": 1 - accuracy_score(y_test, y_pred)`
  Tính tỷ lệ lỗi, bằng 1 trừ accuracy.

- `"precision": precision_score(y_test, y_pred, zero_division=0)`
  Tính precision, nếu xảy ra chia cho 0 thì trả 0 thay vì báo lỗi.

- `"recall": recall_score(y_test, y_pred, zero_division=0)`
  Tính recall.

- `"f1_score": f1_score(y_test, y_pred, zero_division=0)`
  Tính F1-score.

- `"roc_auc": roc_auc_score(y_test, y_prob)`
  Tính ROC-AUC dựa trên xác suất dự đoán.

- `"depth": model.get_depth()`
  Lấy độ sâu của cây.

- `"n_leaves": model.get_n_leaves()`
  Lấy số lá của cây.

- `return result`
  Trả về dictionary chứa toàn bộ kết quả.

- `def show_confusion_matrix(cm, title):`
  Định nghĩa hàm vẽ confusion matrix.

- `disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])`
  Tạo object hiển thị confusion matrix với nhãn lớp.

- `disp.plot(cmap="Blues", values_format="d")`
  Vẽ confusion matrix với tông màu xanh và hiển thị số nguyên.

- `plt.title(title)`
  Đặt tiêu đề cho biểu đồ.

- `plt.grid(False)`
  Tắt lưới nền để hình gọn hơn.

- `plt.show()`
  Hiển thị biểu đồ.

- `def summarize_overfitting(train_acc, test_acc):`
  Định nghĩa hàm nhận xét nhanh về mức độ overfitting.

- `gap = train_acc - test_acc`
  Tính độ chênh giữa accuracy trên train và test.

- `if gap >= 0.10:`
  Nếu chênh lệch từ 10% trở lên thì xem như overfitting khá rõ.

- `return f"..."`
  Trả về câu nhận xét tương ứng.

- `if gap >= 0.05:`
  Nếu chênh lệch từ 5% đến dưới 10% thì xem là overfitting nhẹ.

- `return f"..."`
  Trả về câu nhận xét tương ứng.

- `return f"..."`
  Nếu chênh lệch nhỏ hơn 5% thì xem như chưa overfit quá nặng.

- `def result_to_frame(result):`
  Định nghĩa hàm chuyển dictionary kết quả thành DataFrame.

- `return pd.DataFrame(...)`
  Trả về một bảng kết quả để hiển thị đẹp trong notebook.

- `"Metric": [...]`
  Danh sách tên các chỉ số.

- `"Value": [...]`
  Danh sách giá trị tương ứng lấy từ dictionary `result`.

## Cell 15: Huấn luyện baseline model

```python
baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
baseline_result = evaluate_tree_model(
    "Baseline Decision Tree",
    baseline_model,
    X_train,
    X_test,
    y_train,
    y_test,
)

print("=== BASELINE MODEL RESULTS ===")
display(result_to_frame(baseline_result))

print("Classification report:")
print(classification_report(y_test, baseline_result["y_pred"], zero_division=0))
```

Giải thích từng dòng:

- `baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)`
  Tạo mô hình cây quyết định cơ bản với seed cố định.

- `baseline_result = evaluate_tree_model(...)`
  Gọi hàm đã viết ở trên để train mô hình và thu toàn bộ kết quả đánh giá.

- `"Baseline Decision Tree",`
  Tên dùng để nhận diện mô hình.

- `baseline_model,`
  Truyền mô hình baseline vào hàm.

- `X_train, X_test, y_train, y_test,`
  Truyền dữ liệu train/test vào hàm.

- `print("=== BASELINE MODEL RESULTS ===")`
  In tiêu đề của phần kết quả.

- `display(result_to_frame(baseline_result))`
  Chuyển kết quả thành bảng rồi hiển thị.

- `print("Classification report:")`
  In tiêu đề cho báo cáo phân loại.

- `print(classification_report(y_test, baseline_result["y_pred"], zero_division=0))`
  In báo cáo chi tiết precision, recall, f1-score cho từng lớp.

## Cell 16: Vẽ confusion matrix và cây baseline

```python
# Show the confusion matrix and the top part of the tree.
show_confusion_matrix(
    baseline_result["confusion_matrix"],
    "Confusion Matrix - Baseline Decision Tree"
)

plt.figure(figsize=(24, 12))
plot_tree(
    baseline_result["model"],
    feature_names=X.columns,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3,
)
plt.title("Baseline Decision Tree - Top 3 Levels")
plt.show()
```

Giải thích từng dòng:

- `# Show the confusion matrix and the top part of the tree.`
  Comment cho biết cell này dùng để trực quan kết quả mô hình.

- `show_confusion_matrix(...)`
  Gọi hàm vẽ confusion matrix của mô hình baseline.

- `baseline_result["confusion_matrix"],`
  Lấy ma trận nhầm lẫn đã tính trước đó.

- `"Confusion Matrix - Baseline Decision Tree"`
  Tiêu đề của biểu đồ confusion matrix.

- `plt.figure(figsize=(24, 12))`
  Tạo khung hình lớn để vẽ cây cho rõ.

- `plot_tree(...)`
  Vẽ cấu trúc cây quyết định.

- `baseline_result["model"],`
  Truyền mô hình baseline đã train.

- `feature_names=X.columns,`
  Hiển thị tên thuộc tính tại các nút.

- `class_names=["No Stroke", "Stroke"],`
  Hiển thị tên hai lớp.

- `filled=True,`
  Tô màu các nút theo lớp dự đoán.

- `rounded=True,`
  Làm bo góc các ô để nhìn đẹp hơn.

- `fontsize=9,`
  Đặt cỡ chữ.

- `max_depth=3,`
  Chỉ vẽ 3 tầng đầu để hình không quá rối.

- `plt.title("Baseline Decision Tree - Top 3 Levels")`
  Đặt tiêu đề biểu đồ cây.

- `plt.show()`
  Hiển thị cây.

## Cell 18: Phân tích cây baseline

```python
baseline_tree = baseline_result["model"]
root_feature_index = baseline_tree.tree_.feature[0]
root_threshold = baseline_tree.tree_.threshold[0]
root_feature_name = X.columns[root_feature_index]

feature_importance = (
    pd.Series(baseline_tree.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(10)
)

print("=== BASELINE TREE ANALYSIS ===")
print(f"Tree depth: {baseline_result['depth']}")
print(f"Number of leaves: {baseline_result['n_leaves']}")
print(f"Root split: {root_feature_name} <= {root_threshold:.2f}")
print(summarize_overfitting(baseline_result["train_accuracy"], baseline_result["accuracy"]))

print("\nTop 10 most important features:")
display(feature_importance.to_frame(name="importance"))

print("\nSimplified split rules for the first 4 levels:")
print(export_text(baseline_tree, feature_names=list(X.columns), max_depth=4))
```

Giải thích từng dòng:

- `baseline_tree = baseline_result["model"]`
  Lấy mô hình cây baseline ra một biến riêng để thao tác.

- `root_feature_index = baseline_tree.tree_.feature[0]`
  Lấy chỉ số của thuộc tính dùng để tách ở nút gốc.

- `root_threshold = baseline_tree.tree_.threshold[0]`
  Lấy ngưỡng chia của nút gốc.

- `root_feature_name = X.columns[root_feature_index]`
  Đổi chỉ số thuộc tính thành tên cột thực tế.

- `feature_importance = (...)`
  Tính mức độ quan trọng của từng thuộc tính trong cây.

- `pd.Series(baseline_tree.feature_importances_, index=X.columns)`
  Tạo `Series` với giá trị importance và tên cột tương ứng.

- `.sort_values(ascending=False)`
  Sắp xếp giảm dần từ quan trọng nhất đến ít quan trọng hơn.

- `.head(10)`
  Lấy 10 thuộc tính đứng đầu.

- `print("=== BASELINE TREE ANALYSIS ===")`
  In tiêu đề phần phân tích.

- `print(f"Tree depth: {baseline_result['depth']}")`
  In độ sâu của cây.

- `print(f"Number of leaves: {baseline_result['n_leaves']}")`
  In số lá của cây.

- `print(f"Root split: {root_feature_name} <= {root_threshold:.2f}")`
  In luật tách của nút gốc.

- `print(summarize_overfitting(...))`
  In nhận xét ngắn về việc mô hình có overfit hay không.

- `print("\nTop 10 most important features:")`
  In tiêu đề trước khi hiện bảng importance.

- `display(feature_importance.to_frame(name="importance"))`
  Hiển thị 10 thuộc tính quan trọng nhất dưới dạng bảng.

- `print("\nSimplified split rules for the first 4 levels:")`
  In tiêu đề phần luật tách.

- `print(export_text(baseline_tree, feature_names=list(X.columns), max_depth=4))`
  In cây dưới dạng text để xem các luật chia nhánh ở 4 tầng đầu.

## Cell 19: Nhận xét nhanh từ cây baseline

```python
print("=== Quick takeaways ===")

top3 = feature_importance.head(3).index.tolist()

print(f"The baseline tree reaches a depth of {baseline_result['depth']} with {baseline_result['n_leaves']} leaves.")
print("Train accuracy is noticeably higher than test accuracy, so overfitting is worth watching.")
print(f"The three most influential features here are: {', '.join(top3)}.")
print("Near the top of the tree, age, glucose, and BMI appear early in the split rules.")
print("Since the classes are imbalanced, high accuracy does not automatically mean many stroke cases are being caught.")
```

Giải thích từng dòng:

- `print("=== Quick takeaways ===")`
  In tiêu đề cho phần kết luận ngắn.

- `top3 = feature_importance.head(3).index.tolist()`
  Lấy tên 3 đặc trưng quan trọng nhất và chuyển thành list.

- `print(f"...")`
  Dòng này in nhận xét về độ sâu và số lá của cây baseline.

- `print("Train accuracy is noticeably higher than test accuracy, so overfitting is worth watching.")`
  Nhắc rằng train accuracy cao hơn test khá nhiều, có dấu hiệu overfitting.

- `print(f"... {', '.join(top3)} ...")`
  In ra tên 3 biến quan trọng nhất.

- `print("Near the top of the tree, age, glucose, and BMI appear early in the split rules.")`
  Nhận xét rằng ở các nhánh đầu, age, glucose và BMI xuất hiện sớm.

- `print("Since the classes are imbalanced, high accuracy does not automatically mean many stroke cases are being caught.")`
  Nhấn mạnh rằng accuracy cao chưa chắc bắt được nhiều ca stroke vì dữ liệu lệch lớp.

## Cell 21: Thử nhiều phiên bản mô hình

```python
model_candidates = {
    "Baseline": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Class Weight": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "Pruning": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=5,
        min_samples_leaf=10,
    ),
    "Entropy + Pruning": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=10,
    ),
}

all_results = {}

for model_name, model in model_candidates.items():
    all_results[model_name] = evaluate_tree_model(
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

comparison_rows = []
for model_name, result in all_results.items():
    comparison_rows.append(
        {
            "Model": model_name,
            "Train Accuracy": result["train_accuracy"],
            "Accuracy": result["accuracy"],
            "Error Rate": result["error_rate"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-Score": result["f1_score"],
            "ROC-AUC": result["roc_auc"],
            "Depth": result["depth"],
            "Leaves": result["n_leaves"],
        }
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Accuracy", ascending=False)
display(comparison_df.round(4))
```

Giải thích từng dòng:

- `model_candidates = { ... }`
  Tạo dictionary chứa nhiều phiên bản Decision Tree để so sánh.

- `"Baseline": DecisionTreeClassifier(random_state=RANDOM_STATE),`
  Mô hình cây mặc định.

- `"Class Weight": DecisionTreeClassifier(...)`
  Mô hình có `class_weight="balanced"` để mô hình chú ý hơn đến lớp thiểu số.

- `random_state=RANDOM_STATE,`
  Cố định random seed.

- `class_weight="balanced",`
  Tự động cân bằng trọng số lớp dựa trên tần suất.

- `"Pruning": DecisionTreeClassifier(...)`
  Mô hình có cắt tỉa cây bằng cách giới hạn độ sâu và số mẫu tối thiểu ở lá.

- `max_depth=5,`
  Giới hạn cây chỉ sâu tối đa 5 tầng.

- `min_samples_leaf=10,`
  Mỗi nút lá phải có ít nhất 10 mẫu.

- `"Entropy + Pruning": DecisionTreeClassifier(...)`
  Mô hình dùng tiêu chí `entropy` thay cho mặc định và vẫn cắt tỉa cây.

- `criterion="entropy",`
  Dùng entropy để chọn cách chia nhánh.

- `all_results = {}`
  Tạo dictionary rỗng để lưu kết quả của từng mô hình.

- `for model_name, model in model_candidates.items():`
  Lặp qua từng mô hình trong danh sách.

- `all_results[model_name] = evaluate_tree_model(...)`
  Huấn luyện mô hình và lưu kết quả theo tên mô hình.

- `comparison_rows = []`
  Tạo list rỗng để sau đó gom thành bảng so sánh.

- `for model_name, result in all_results.items():`
  Lặp qua kết quả của từng mô hình.

- `comparison_rows.append({ ... })`
  Thêm một dòng dữ liệu vào danh sách so sánh.

- `"Model": model_name`
  Tên mô hình.

- `"Train Accuracy": result["train_accuracy"]`
  Accuracy trên train.

- `"Accuracy": result["accuracy"]`
  Accuracy trên test.

- `"Error Rate": result["error_rate"]`
  Tỷ lệ lỗi.

- `"Precision": result["precision"]`
  Precision.

- `"Recall": result["recall"]`
  Recall.

- `"F1-Score": result["f1_score"]`
  F1-score.

- `"ROC-AUC": result["roc_auc"]`
  ROC-AUC.

- `"Depth": result["depth"]`
  Độ sâu của cây.

- `"Leaves": result["n_leaves"]`
  Số lá của cây.

- `comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Accuracy", ascending=False)`
  Chuyển danh sách thành DataFrame và sắp xếp theo accuracy giảm dần.

- `display(comparison_df.round(4))`
  Hiển thị bảng kết quả và làm tròn 4 chữ số thập phân.

## Cell 22: Vẽ biểu đồ so sánh các mô hình

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.barplot(data=comparison_df, x="Accuracy", y="Model", palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Accuracy comparison")

sns.barplot(data=comparison_df, x="Recall", y="Model", palette="magma", ax=axes[0, 1])
axes[0, 1].set_title("Recall comparison")

sns.barplot(data=comparison_df, x="F1-Score", y="Model", palette="crest", ax=axes[1, 0])
axes[1, 0].set_title("F1-score comparison")

sns.barplot(data=comparison_df, x="ROC-AUC", y="Model", palette="rocket", ax=axes[1, 1])
axes[1, 1].set_title("ROC-AUC comparison")

plt.tight_layout()
plt.show()
```

Giải thích từng dòng:

- `fig, axes = plt.subplots(2, 2, figsize=(16, 10))`
  Tạo khung 4 biểu đồ để so sánh các chỉ số.

- `sns.barplot(data=comparison_df, x="Accuracy", y="Model", palette="viridis", ax=axes[0, 0])`
  Vẽ biểu đồ cột so sánh accuracy giữa các mô hình.

- `axes[0, 0].set_title("Accuracy comparison")`
  Đặt tiêu đề cho biểu đồ accuracy.

- `sns.barplot(data=comparison_df, x="Recall", y="Model", palette="magma", ax=axes[0, 1])`
  Vẽ biểu đồ so sánh recall.

- `axes[0, 1].set_title("Recall comparison")`
  Đặt tiêu đề cho biểu đồ recall.

- `sns.barplot(data=comparison_df, x="F1-Score", y="Model", palette="crest", ax=axes[1, 0])`
  Vẽ biểu đồ so sánh F1-score.

- `axes[1, 0].set_title("F1-score comparison")`
  Đặt tiêu đề cho biểu đồ F1-score.

- `sns.barplot(data=comparison_df, x="ROC-AUC", y="Model", palette="rocket", ax=axes[1, 1])`
  Vẽ biểu đồ so sánh ROC-AUC.

- `axes[1, 1].set_title("ROC-AUC comparison")`
  Đặt tiêu đề cho biểu đồ ROC-AUC.

- `plt.tight_layout()`
  Căn lại khoảng cách giữa các biểu đồ.

- `plt.show()`
  Hiển thị hình.

## Cell 23: Chọn mô hình tốt nhất theo từng chỉ số

```python
best_accuracy_model = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
best_f1_model = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
best_recall_model = comparison_df.sort_values(by="Recall", ascending=False).iloc[0]["Model"]

print("=== Quick comparison ===")
print(f"Highest accuracy: {best_accuracy_model}")
print(f"Highest F1-score: {best_f1_model}")
print(f"Highest recall: {best_recall_model}")
print("")
print("If the goal is to catch more stroke cases, recall and ROC-AUC deserve more attention than accuracy alone.")
```

Giải thích từng dòng:

- `best_accuracy_model = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]`
  Sắp xếp bảng theo accuracy giảm dần rồi lấy tên mô hình đứng đầu.

- `best_f1_model = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]`
  Lấy mô hình có F1-score cao nhất.

- `best_recall_model = comparison_df.sort_values(by="Recall", ascending=False).iloc[0]["Model"]`
  Lấy mô hình có recall cao nhất.

- `print("=== Quick comparison ===")`
  In tiêu đề phần so sánh nhanh.

- `print(f"Highest accuracy: {best_accuracy_model}")`
  In tên mô hình có accuracy cao nhất.

- `print(f"Highest F1-score: {best_f1_model}")`
  In tên mô hình có F1-score cao nhất.

- `print(f"Highest recall: {best_recall_model}")`
  In tên mô hình có recall cao nhất.

- `print("")`
  In một dòng trống cho dễ đọc.

- `print("If the goal is to catch more stroke cases, recall and ROC-AUC deserve more attention than accuracy alone.")`
  Nhắc rằng nếu mục tiêu là phát hiện ca stroke thì recall và ROC-AUC quan trọng hơn chỉ nhìn accuracy.

## Cell 25: Vẽ cây đã cải tiến

```python
improved_tree_result = all_results["Entropy + Pruning"]

show_confusion_matrix(
    improved_tree_result["confusion_matrix"],
    "Confusion Matrix - Entropy + Pruning"
)

plt.figure(figsize=(22, 10))
plot_tree(
    improved_tree_result["model"],
    feature_names=X.columns,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3,
)
plt.title("Improved Decision Tree - Entropy + Pruning (Top 3 Levels)")
plt.show()
```

Giải thích từng dòng:

- `improved_tree_result = all_results["Entropy + Pruning"]`
  Lấy kết quả của mô hình `Entropy + Pruning` từ dictionary `all_results`.

- `show_confusion_matrix(...)`
  Vẽ confusion matrix cho mô hình cải tiến này.

- `improved_tree_result["confusion_matrix"],`
  Truyền ma trận nhầm lẫn của mô hình cải tiến.

- `"Confusion Matrix - Entropy + Pruning"`
  Tiêu đề của confusion matrix.

- `plt.figure(figsize=(22, 10))`
  Tạo khung hình để vẽ cây.

- `plot_tree(...)`
  Vẽ cây quyết định của mô hình cải tiến.

- `improved_tree_result["model"],`
  Lấy mô hình đã train để vẽ.

- `feature_names=X.columns,`
  Hiển thị tên đặc trưng trên cây.

- `class_names=["No Stroke", "Stroke"],`
  Hiển thị tên hai lớp.

- `filled=True,`
  Tô màu theo lớp.

- `rounded=True,`
  Bo góc các nút.

- `fontsize=9,`
  Đặt cỡ chữ.

- `max_depth=3,`
  Chỉ vẽ 3 tầng đầu để cây dễ đọc.

- `plt.title("Improved Decision Tree - Entropy + Pruning (Top 3 Levels)")`
  Đặt tiêu đề cho biểu đồ cây cải tiến.

- `plt.show()`
  Hiển thị hình.

## Tóm tắt luồng chạy của notebook

- Cell 2: import thư viện và cấu hình.
- Cell 4, 5: đọc dữ liệu và xem tổng quan.
- Cell 7: kiểm tra missing values, duplicate và class imbalance.
- Cell 9: vẽ biểu đồ khám phá dữ liệu.
- Cell 11: tiền xử lý và chia train/test.
- Cell 13: tạo các hàm hỗ trợ đánh giá.
- Cell 15, 16: huấn luyện baseline và trực quan kết quả.
- Cell 18, 19: phân tích cây baseline.
- Cell 21, 22, 23: thử các phiên bản khác và so sánh.
- Cell 25: vẽ cây cải tiến để đối chiếu với baseline.
