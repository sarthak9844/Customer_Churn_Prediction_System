import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from src.data_cleaning import load_data, clean_data


# 1. Load saved model
model = joblib.load(r"D:\PROJECT\Customer_Churn_Prediction_System\models\tuned_random_model.pkl")

print("Model loaded successfully.")


# 2. Load and prepare data
file_path = r"D:\PROJECT\Customer_Churn_Prediction_System\data\Telco-Customer-Churn.csv"

df = load_data(file_path)
df = clean_data(df)



# 3. Separate features and target
X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# 4. Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()

plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    r"images\confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved.")


# 7. ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Customer Churn - ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    r"images\roc_curve.png",
    dpi=300
)

plt.close()

print(f"ROC-AUC: {roc_auc:.4f}")
print("ROC curve saved.")