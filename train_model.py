import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os

# 1. Load data
path = "./dataset/Arabic Sign Language Letters Dataset.csv"
if not os.path.exists(path):
    print(f"Error: File not found at {path}")
    exit(1)

df = pd.read_csv(path)
print("Data loaded. Shape:", df.shape)

# 2.1 Separate features and labels
X = df.drop(columns=["letter"])
y = df["letter"]

# 2.2 Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded classes:", le.classes_)

# 2.3 Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 2.4 Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to train
models = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

best_accuracy = 0
best_model = None
best_model_name = ""

print("\n--- Starting Training & Evaluation ---")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    print(f"{name} Accuracy: {acc:.4f}")
    
    # Save report to a separate file
    report_filename = f"report_{name.lower()}.txt"
    with open(report_filename, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved to {report_filename}")
    
    # Track the best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n--- Comparison Complete ---")
print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model and remaining artifacts
joblib.dump(best_model, "hand_sign_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# Also save a summary file
with open("classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Accuracy: {best_accuracy:.4f}\n\n")
    f.write(f"Detailed report for {best_model_name}:\n")
    y_pred_best = best_model.predict(X_test_scaled)
    f.write(classification_report(y_test, y_pred_best, target_names=le.classes_))

print("Best model and artifacts saved successfully.")
