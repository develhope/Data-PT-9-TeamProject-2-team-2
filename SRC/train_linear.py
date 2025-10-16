#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Linear Regression (baseline) sul dataset Car Sales.

Output prodotti (tutti sotto data/processed/):
- splits/<split_name>.csv                         → indici train/test per garantire stesso split in tutti i modelli
- models/linear_model_pipeline.joblib             → pipeline + modello addestrato (pronto per inferenza)
- models/linear_metrics.json                      → metriche di valutazione (MAE, RMSE, R2)
- models/linear_coefficients.csv                  → coefficenti del modello (importanza interpretabile)
- predictions/linear_pred_vs_actual_test.csv      → confronto actual vs predicted sul test set
- figures/linear_predicted_vs_actual.png          → scatter Predetto vs Reale (diagnostico veloce)
"""

from __future__ import annotations

# --- librerie standard ---
import argparse
import json
from pathlib import Path

# --- librerie scientifiche ---
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- scikit-learn: preprocessing, split, modello e metriche ---
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1) Utility: trovare la root del repo in modo "robusto"
#    (così i path restano relativi e non rompono su altri PC)
# ------------------------------------------------------------
def get_repo_root() -> Path:
    here = Path.cwd().resolve()
    for base in (here, here.parent, here.parent.parent):
        if (base / "data").exists():
            return base
    return here  # fallback: se non trova "data", usa la working dir corrente


# ------------------------------------------------------------
# 2) Argomenti terminale (così possiamo riusare lo script da terminale)
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Linear Regression su Car Sales (baseline interpretabile)")
    p.add_argument("--test-size", type=float, default=0.2, help="Quota di test set (es. 0.2 = 20%)")
    p.add_argument("--random-state", type=int, default=42, help="Seed per riproducibilità dello split")
    p.add_argument("--split-name", type=str, default="split_v1", help="Nome file per salvare gli indici di split")
    return p.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------
    # 3) Definizione cartelle d'uscita (tutte in data/processed)
    # --------------------------------------------------------
    REPO_ROOT = get_repo_root()
    PROC_DIR  = REPO_ROOT / "data" / "processed"
    RAW_FILE  = PROC_DIR / "database_cleaned_2.csv"   # sorgente già pulita

    OUT_SPLITS = PROC_DIR / "splits"
    OUT_MODELS = PROC_DIR / "models"
    OUT_PREDS  = PROC_DIR / "predictions"
    OUT_FIGS   = PROC_DIR / "figures"

    for p in [OUT_SPLITS, OUT_MODELS, OUT_PREDS, OUT_FIGS]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Dataset  : {RAW_FILE}")
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"File mancante: {RAW_FILE}. Generarlo prima con i notebook di cleaning.")

    # --------------------------------------------------------
    # 4) Carico il dataset
    # --------------------------------------------------------
    df = pd.read_csv(RAW_FILE, low_memory=False)
    print(f"[INFO] Shape iniziale: {df.shape}")

    # --------------------------------------------------------
    # 5) Scelgo target e feature
    #    - Target: Price 
    #    - Feature: mix di numeriche e categoriche
    # --------------------------------------------------------

    # Target fisso: nel nostro cleaned abbiamo 'Price' (no 'Price ($)')
    TARGET = "Price"
    if TARGET not in df.columns:
        raise ValueError(
            f"Colonna target '{TARGET}' non trovata. Colonne disponibili: {list(df.columns)}"
        )

    # Candidiamo alcune feature informative 
    candidate_features = [
        "Annual Income",     # numerica
        "Company",           # categorica
        "Model",             # categorica
        "Transmission",      # categorica
        "Dealer_Region",     # categorica
        "Body Style",        # categorica (se presente)
    ]

    # Intersezione con lo schema reale del dataset (evita rotture se manca qualcosa)
    feature_cols = [c for c in candidate_features if c in df.columns]
    if not feature_cols:
        raise ValueError(
            "Nessuna feature disponibile tra le candidate. Controllare lo schema dati."
        )

    # X = feature, y = target (forzo numerico)
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    # Patch: se ci fossero target NaN (per qualsiasi motivo), li escludo
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # --------------------------------------------------------
    # 6) Split train/test CONDIVISO
    #    - Gli indici salvati permettono anche a Matteo (per RandomForest)
    #      di usare esattamente lo stesso split per confronto corretto.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    split_path = OUT_SPLITS / f"{args.split_name}.csv"

    # Salvo gli indici: una riga per indice, con colonna split=train/test
    split_train = pd.DataFrame({"row_index": X_train.index, "split": "train"})
    split_test  = pd.DataFrame({"row_index": X_test.index,  "split": "test"})
    split_df = pd.concat([split_train, split_test], ignore_index=True)

    split_df.to_csv(split_path, index=False)
    print(f"[INFO] Split salvato → {split_path.name} (righe: {len(split_df)})")

    # --------------------------------------------------------
    # 7) Preprocessing con ColumnTransformer + Pipeline
    #    - Dato che l'EDA/cleaning ha già gestito i missing/outlier,
    #      NON usiamo imputazione né scaling.
    #    - Categoriche: One-Hot (handle_unknown="ignore" per robustezza)
    #    - Numeriche: le lasciamo "as is" (remainder='passthrough')
    #    - Modello: LinearRegression (baseline interpretabile)
    # --------------------------------------------------------

    num_cols = [c for c in ["Annual Income"] if c in X_train.columns]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"  # tiene le numeriche così come sono
    )

    model = Pipeline([
        ("preproc", preproc),
        ("regressor", LinearRegression())
    ])

    print(f"[INFO] Feature numeriche   : {num_cols}")
    print(f"[INFO] Feature categoriche : {cat_cols}")

    # --------------------------------------------------------
    # 8) Training
    # --------------------------------------------------------
    model.fit(X_train, y_train)
    print("[INFO] Modello addestrato.")

    # --------------------------------------------------------
    # 9) Valutazione su test (metriche standard)
    #    - MAE  : errore medio assoluto (in €)
    #    - RMSE : radice MSE (penalizza di più gli errori grossi)
    #    - R2   : varianza spiegata (0..1, più alto è meglio)
    # --------------------------------------------------------
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))

    # Per versioni vecchie di sklearn: niente parametro 'squared'
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    r2  = float(r2_score(y_test, y_pred))

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    metrics_path = OUT_MODELS / "linear_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metriche → {metrics_path.name}: {metrics}")

    # --------------------------------------------------------
    # 10) Salvataggi: modello e predizioni
    #    - Salvo la pipeline completa (preprocessing + regressore) pronta per inferenza.
    #    - Predizioni: allineo sempre a NumPy e verifico che le lunghezze coincidano,
    #      così evitiamo errori tipo "All arrays must be of the same length".
    # --------------------------------------------------------
    model_path = OUT_MODELS / "linear_model_pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"[INFO] Modello salvato → {model_path.name}")

    # Converto a numpy per evitare problemi di indice/forma
    y_test_np = np.asarray(y_test)
    y_pred_np = np.asarray(y_pred)

    # Safety check: stessa lunghezza
    assert len(y_test_np) == len(y_pred_np), (
        f"Len mismatch tra y_test ({len(y_test_np)}) e y_pred ({len(y_pred_np)})"
    )

    # Costruisco il dataframe di confronto (reale vs predetto)
    pred_df = pd.DataFrame({
        "actual":   y_test_np,
        "predicted": y_pred_np
    })

    pred_path = OUT_PREDS / "linear_pred_vs_actual_test.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predizioni test → {pred_path.name}")

    # --------------------------------------------------------
    # 11) Coefficienti (interpretabilità)
    #     - Prendo i nomi direttamente dal ColumnTransformer
    #       per evitare mismatch di lunghezze.
    # --------------------------------------------------------
    pre = model.named_steps["preproc"]

    # sklearn moderno: ColumnTransformer.get_feature_names_out()
    try:
        feature_names = pre.get_feature_names_out()
    except AttributeError:
        # Fallback per versioni vecchie: ricostruisco manualmente
        # 1) nomi OHE
        ohe = pre.named_transformers_["cat"]
        cat_cols = [c for c in X_train.columns if c not in ["Annual Income"]]
        cat_names = ohe.get_feature_names_out(cat_cols)
        # 2) numeriche (passthrough alla fine)
        num_cols = [c for c in ["Annual Income"] if c in X_train.columns]
        feature_names = np.concatenate([cat_names, np.array(num_cols, dtype=object)])

    coefs = model.named_steps["regressor"].coef_

    # Sanity check: stessa lunghezza
    if len(coefs) != len(feature_names):
        print("[WARN] Lunghezze diverse: coefs:", len(coefs), " feature_names:", len(feature_names))
        # Come ultima difesa, tronco alla minima lunghezza per non far esplodere il salvataggio
        m = min(len(coefs), len(feature_names))
        coefs = coefs[:m]
        feature_names = feature_names[:m]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    coef_path = OUT_MODELS / "linear_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"[INFO] Coefficienti → {coef_path.name}")
    # --------------------------------------------------------
    # 12) Grafico Predicted vs Actual (diagnostico)
    # --------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    lims = [min(np.nanmin(y_test), y_pred.min()), max(np.nanmax(y_test), y_pred.max())]
    plt.plot(lims, lims, "--", color="red")  # linea di identità (pred=actual)
    plt.xlabel("Prezzo reale")
    plt.ylabel("Prezzo predetto")
    plt.title("Linear Regression — Predicted vs Actual")
    plt.tight_layout()
    fig_path = OUT_FIGS / "linear_predicted_vs_actual.png"
    plt.savefig(fig_path, dpi=120)
    print(f"[INFO] Figura → {fig_path.name}")


if __name__ == "__main__":
    main()