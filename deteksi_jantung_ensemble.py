import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
DATA_PATH = Path("data/heart.csv")


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)


def load_dataset(path: Path = DATA_PATH, url: str = DATA_URL) -> Tuple[pd.DataFrame, str]:
    """
    Mengembalikan DataFrame dengan fitur klinis.
    - Prioritas membaca file lokal.
    - Jika belum ada, otomatis mengunduh dari URL resmi lalu menyimpannya.
    - Label encoding untuk kolom bertipe kategori (mis. 'thal').
    """
    source = "local"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        source = "downloaded"

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    # Standarisasi nama kolom target
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan pada dataset.")

    # Label encoding untuk fitur kategorikal
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df, source


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split stratified, scaling, dan SMOTE hanya pada data latih.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    return X_res, X_test_scaled, y_res, y_test, scaler


def build_base_models(seed: int = SEED) -> Dict[str, object]:
    """
    Definisi base learners. Parameter dibuat moderat agar stabil untuk data tabular klinis.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=600, random_state=seed, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf", probability=True, random_state=seed, class_weight="balanced"
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=800, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            use_label_encoder=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        ),
    }


def compute_metrics(y_true, y_pred, proba=None, train_time=None) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, proba) if proba is not None else np.nan

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F2": f2,
        "AUC": auc,
        "Train_Time(s)": train_time if train_time is not None else np.nan,
    }


def fit_and_evaluate(
    name: str, model, X_train, y_train, X_test, y_test
) -> Tuple[Dict[str, float], object]:
    start = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - start
    y_pred = model.predict(X_test)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)

    metrics = compute_metrics(y_test, y_pred, proba, train_time)
    return metrics, model


def evaluate_base_models(
    models: Dict[str, object], X_train, y_train, X_test, y_test
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    results = {}
    fitted = {}
    for name, model in models.items():
        metrics, fitted_model = fit_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        results[name] = metrics
        fitted[name] = fitted_model
        print(
            f"{name:<18} | F1: {metrics['F1']:.4f} | Recall: {metrics['Recall']:.4f} | "
            f"Acc: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f}"
        )
    df = pd.DataFrame(results).T.sort_values(by="F1", ascending=False)
    return df, fitted


def hard_voting(trained: Dict[str, object], X_test) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard voting dari tiga model terbaik (XGB, LightGBM, SVM).
    Proba dihitung sebagai rata-rata proba base learners agar AUC tetap tersedia.
    """
    selected = [trained["XGBoost"], trained["LightGBM"], trained["SVM (RBF Kernel)"]]
    pred_matrix = np.column_stack([m.predict(X_test) for m in selected])
    hard_pred = (pred_matrix.sum(axis=1) >= 2).astype(int)
    prob_matrix = np.column_stack([m.predict_proba(X_test)[:, 1] for m in selected])
    avg_prob = prob_matrix.mean(axis=1)
    return hard_pred, avg_prob


def weighted_average(trained: Dict[str, object], X_test, weights=(0.4, 0.3, 0.3)):
    m1 = trained["XGBoost"]
    m2 = trained["LightGBM"]
    m3 = trained["SVM (RBF Kernel)"]

    prob = (
        m1.predict_proba(X_test)[:, 1] * weights[0]
        + m2.predict_proba(X_test)[:, 1] * weights[1]
        + m3.predict_proba(X_test)[:, 1] * weights[2]
    )
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def stacking_classifier(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    estimators = [
        ("xgb", XGBClassifier(
            n_estimators=250,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            use_label_encoder=False,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=250,
            learning_rate=0.07,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            verbose=-1,
        )),
        ("svm", SVC(kernel="rbf", probability=True, random_state=SEED, class_weight="balanced")),
        ("rf", RandomForestClassifier(
            n_estimators=250, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"
        )),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=400, class_weight="balanced"),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    metrics, model = fit_and_evaluate("Stacking", stacking, X_train, y_train, X_test, y_test)
    return metrics, model


def build_report_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    ordered_cols = ["Accuracy", "Precision", "Recall", "F1", "F2", "AUC", "Train_Time(s)"]
    return df[ordered_cols].sort_values(by="F1", ascending=False)


def print_reference_report():
    """
    Tabel hasil referensi (sesuai laporan penelitian) untuk memudahkan perbandingan.
    Angka berikut bersifat ilustratif dari eksperimen terdahulu.
    """
    data = {
        "Model/Ensemble": [
            "Logistic Regression (Baseline)",
            "Random Forest (Baseline)",
            "SVM (RBF Kernel)",
            "XGBoost",
            "LightGBM",
            "Ensemble: Hard Voting",
            "Ensemble: Weighted Avg",
            "Ensemble: Stacking (LR)",
        ],
        "Akurasi (%)": [78.50, 90.50, 84.00, 92.00, 93.00, 88.00, 89.50, 95.50],
        "Precision (%)": [74.00, 88.00, 80.50, 90.50, 91.50, 86.50, 88.00, 94.00],
        "Recall (%)": [70.50, 89.00, 82.00, 91.50, 92.00, 87.50, 89.00, 94.50],
        "F1-Score (%)": [72.20, 88.50, 81.20, 91.00, 91.70, 87.00, 88.50, 94.20],
        "F2-Score (%)": [71.00, 88.80, 81.70, 91.30, 91.90, 87.30, 88.80, 94.40],
        "AUC": [0.840, 0.945, 0.880, 0.965, 0.975, 0.930, 0.940, 0.985],
        "Waktu Pelatihan (s)": [0.15, 1.20, 3.50, 0.75, 0.40, 3.80, 4.00, 4.50],
    }
    ref_df = pd.DataFrame(data)
    print("\n=== Referensi Hasil (Laporan) ===")
    print(ref_df.to_string(index=False))


def main():
    seed_everything(SEED)
    print("=== [1] Memuat data ===")
    df, source = load_dataset()
    print(f"Dataset source : {source} ({len(df)} baris)")
    print(f"Distribusi awal: {df['target'].value_counts().to_dict()}")

    print("\n=== [2] Preprocessing (Split, Scaling, SMOTE) ===")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print(f"Distribusi setelah SMOTE: {dict(pd.Series(y_train).value_counts())}")

    print("\n=== [3] Evaluasi Base Learners ===")
    base_models = build_base_models()
    base_df, trained_models = evaluate_base_models(base_models, X_train, y_train, X_test, y_test)
    print("\nRingkasan Base Learners (diurutkan F1):")
    print(base_df.round(4))

    print("\n=== [4] Ensemble Learning ===")
    # Hard Voting
    hv_pred, hv_prob = hard_voting(trained_models, X_test)
    hv_metrics = compute_metrics(y_test, hv_pred, hv_prob, train_time=None)

    # Weighted Average
    wa_pred, wa_prob = weighted_average(trained_models, X_test)
    wa_metrics = compute_metrics(y_test, wa_pred, wa_prob, train_time=None)

    # Stacking
    stack_metrics, stack_model = stacking_classifier(X_train, y_train, X_test, y_test)

    ensemble_results = {
        "Hard Voting": hv_metrics,
        "Weighted Avg": wa_metrics,
        "Stacking (LR)": stack_metrics,
    }

    ensemble_df = build_report_table(ensemble_results)
    print("\nRingkasan Ensemble:")
    print(ensemble_df.round(4))

    print("\n=== [5] Perbandingan Terbaik ===")
    best_base_name = base_df.index[0]
    best_base_metrics = base_df.loc[best_base_name]
    compare = pd.DataFrame(
        {
            f"{best_base_name} (Base)": best_base_metrics,
            "Stacking (LR) (Ensemble)": pd.Series(stack_metrics),
        }
    )
    print(compare.round(4))

    print_reference_report()
    print("\nSelesai. Pipeline end-to-end telah dijalankan.")


if __name__ == "__main__":
    main()
