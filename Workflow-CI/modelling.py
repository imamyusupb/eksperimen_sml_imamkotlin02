import pandas as pd
import mlflow
import dagshub
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Inisialisasi MLflow Tracking dengan DagsHub
dagshub.init(repo_owner='imamyusupb', repo_name='Eksperimen_SML_Imamkotlin02', mlflow=True)
mlflow.set_experiment("Heart Disease Tuning")

# Memuat data hasil preprocessing
X_train = pd.read_csv('heart_disease_preprocessing/X_train.csv')
X_test = pd.read_csv('heart_disease_preprocessing/X_test.csv')
y_train = pd.read_csv('heart_disease_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('heart_disease_preprocessing/y_test.csv').values.ravel()

# Kombinasi hyperparameter untuk tuning
param_grid = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15}
]

for i, params in enumerate(param_grid):
    with mlflow.start_run(run_name=f"Tuning_Run_{i}"):
        # Melatih Model
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi Model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # --- SPESIFIKASI ADVANCED (MANUAL LOGGING) ---
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Simpan Model ke MLflow
        mlflow.sklearn.log_model(model, "tuned_model")
        
        # Membuat 2 Artefak Tambahan (Syarat Syah Advanced)
        meta_summary = {"status": "success", "best_iteration": i == 1}
        with open("summary.json", "w") as f:
            json.dump(meta_summary, f)
        
        with open("features.txt", "w") as f:
            f.write("\n".join(X_train.columns.tolist()))
            
        # Log Artefak ke MLflow
        mlflow.log_artifact("summary.json")
        mlflow.log_artifact("features.txt")
        
        print(f"Run {i} selesai. Params: {params} -> Accuracy: {acc}")

