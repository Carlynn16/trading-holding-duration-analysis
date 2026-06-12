"""
Predictive modeling: can a trade's eventual loss be predicted from entry-time
information only?

Deployable features (observable when the trade is opened):
  is_buy, log_volume_std, sym_<X> (top-8 one-hot), hour_sin/cos,
  dow_sin/cos, month_sin/cos.

Forbidden features (known only at or after close):
  Profit, Price.1, Time.1, Volume.1, holding_duration_hours/days,
  duration bin columns, is_trimmed_95, log_profit.

The include_duration=True variant of build_features adds holding_duration_hours;
that variant is DIAGNOSTIC ONLY — it quantifies the ceiling a non-deployable
feature would unlock, motivating the survival-based exit rule from Section 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TOP_SYMBOLS = 8

FORBIDDEN_COLS = frozenset({
    "Profit", "Price.1", "Volume.1", "Time.1",
    "holding_duration_hours", "holding_duration_days",
    "duration_bin_hours", "duration_bin_days",
    "is_trimmed_95", "log_profit",
})


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    include_duration: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y from a trades DataFrame.

    include_duration=False  → deployable entry-time features only.
    include_duration=True   → adds holding_duration_hours (DIAGNOSTIC ONLY).
    """
    result = pd.DataFrame(index=df.index)

    result["is_buy"] = (df["Type"].astype(str) == "Buy").astype(np.int8)

    log_vol = np.log1p(df["Volume"].astype(float))
    mu, sigma = float(log_vol.mean()), float(log_vol.std())
    result["log_volume_std"] = ((log_vol - mu) / max(sigma, 1e-9)).astype(np.float32)

    top_syms = df["Symbol"].value_counts().head(TOP_SYMBOLS).index.tolist()
    for sym in top_syms:
        result[f"sym_{sym}"] = (df["Symbol"].astype(str) == sym).astype(np.int8)

    hour = (
        df["hour"].astype(float)
        if "hour" in df.columns
        else pd.to_datetime(df["Time"]).dt.hour.astype(float)
    )
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    dow = (
        df["dayofweek"].astype(float)
        if "dayofweek" in df.columns
        else pd.to_datetime(df["Time"]).dt.dayofweek.astype(float)
    )
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    if "month" in df.columns:
        month = df["month"].astype(float)
    elif "Time" in df.columns:
        month = pd.to_datetime(df["Time"]).dt.month.astype(float)
    else:
        month = None

    if month is not None:
        result["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
        result["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)

    if include_duration:
        result["holding_duration_hours"] = df["holding_duration_hours"].values

    y = df["is_loss"].astype(int)
    return result.reset_index(drop=True), y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_models(spw: float, seed: int) -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=seed,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Train and compare
# ---------------------------------------------------------------------------

def train_compare(
    df: pd.DataFrame,
    sample_n: int = 500_000,
    seed: int = 42,
) -> dict:
    """Train 4 classifiers on entry-time features; return metrics + fitted objects.

    Returns
    -------
    comparison      – DataFrame (model, roc_auc, pr_auc, precision, recall, f1)
    best_model_name – str
    best_model      – fitted estimator
    models          – {name: fitted estimator}
    X_test, y_test  – held-out test set
    probas          – {name: probability array on test set}
    cv_auc          – 5-fold stratified CV ROC-AUC of best model (on 100K subset)
    feature_names   – list of column names in X
    """
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed)
    X, y = build_features(sample, include_duration=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    model_defs = _make_models(spw=spw, seed=seed)
    models_fitted: dict = {}
    probas: dict = {}
    rows = []

    for name, model in model_defs.items():
        print(f"    {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        models_fitted[name] = model
        probas[name] = y_prob
        rows.append({
            "model"    : name,
            "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc"   : round(average_precision_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name  = comparison.iloc[0]["model"]
    best_model = models_fitted[best_name]

    # 5-fold CV on a 100K subsample of the training set (avoids multi-hour CV run)
    cv_n = min(100_000, len(X_train))
    rng  = np.random.default_rng(seed)
    cv_idx = rng.choice(len(X_train), cv_n, replace=False)
    X_cv = X_train.iloc[cv_idx]
    y_cv = y_train.iloc[cv_idx]
    best_fresh = _make_models(spw=spw, seed=seed)[best_name]
    cv_auc = float(cross_val_score(
        best_fresh, X_cv, y_cv,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="roc_auc",
        n_jobs=1,
    ).mean())

    return dict(
        comparison=comparison,
        best_model_name=best_name,
        best_model=best_model,
        models=models_fitted,
        X_test=X_test,
        y_test=y_test,
        probas=probas,
        cv_auc=cv_auc,
        feature_names=list(X_train.columns),
    )


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def leakage_check(
    df: pd.DataFrame,
    sample_n: int = 200_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compare entry-only vs entry+duration ROC-AUC (LightGBM, fixed params).

    Returns (auc_entry_only, auc_entry_plus_duration).
    Duration is a post-close observation — not deployable live.
    The gap quantifies what the survival-based exit rule captures that an
    entry classifier cannot.
    """
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed)
    aucs = []

    for include_dur in [False, True]:
        X, y = build_features(sample, include_duration=include_dur)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        spw = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
        model = LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.05,
            scale_pos_weight=spw,
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        aucs.append(float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])))

    return aucs[0], aucs[1]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_values(best_model, X_sample: pd.DataFrame):
    """Compute SHAP values for best_model on X_sample.

    Handles Pipeline (LR) and tree models (RF, XGB, LGBM).
    Returns a SHAP Explanation object with .values shape (n, n_features).
    """
    import shap

    if isinstance(best_model, Pipeline):
        X_t = pd.DataFrame(
            best_model[:-1].transform(X_sample),
            columns=X_sample.columns,
            index=X_sample.index,
        )
        inner = list(best_model.named_steps.values())[-1]
        exp = shap.LinearExplainer(inner, X_t)
        sv = exp(X_t)
        sv.feature_names = list(X_sample.columns)
    else:
        exp = shap.TreeExplainer(best_model)
        sv  = exp(X_sample, check_additivity=False)
        # RF returns 3-D [n, features, 2] for binary; take class-1 slice
        if sv.values.ndim == 3:
            sv = shap.Explanation(
                values=sv.values[:, :, 1],
                base_values=(
                    sv.base_values[:, 1]
                    if sv.base_values.ndim > 1
                    else sv.base_values
                ),
                data=sv.data,
                feature_names=sv.feature_names,
            )
    return sv


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def save_modeling_outputs(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
) -> dict:
    """Run full modeling pipeline, persist CSVs, return all fitted objects."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Training 4 models (entry-only features)...")
    results = train_compare(df)
    results["comparison"].to_csv(out / "model_comparison.csv", index=False)
    print(
        f"  Best: {results['best_model_name']}"
        f"  ROC-AUC={results['comparison'].iloc[0]['roc_auc']:.4f}"
        f"  5-fold CV={results['cv_auc']:.4f}"
    )

    print("  Leakage check (entry-only vs entry+duration)...")
    auc_entry, auc_dur = leakage_check(df)
    pd.DataFrame([{
        "model"                  : "LightGBM (fixed params)",
        "auc_entry_only"         : round(auc_entry, 4),
        "auc_entry_plus_duration": round(auc_dur, 4),
    }]).to_csv(out / "leakage_auc.csv", index=False)
    print(f"  Entry-only: {auc_entry:.4f}  |  Entry+duration: {auc_dur:.4f}")

    print("  SHAP values (2 000-row sample)...")
    X_test     = results["X_test"]
    shap_sample = X_test.sample(n=min(2_000, len(X_test)), random_state=42)
    sv = shap_values(results["best_model"], shap_sample)
    shap_df = (
        pd.DataFrame({
            "feature"      : list(shap_sample.columns),
            "mean_abs_shap": np.abs(sv.values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(out / "shap_top_features.csv", index=False)

    return dict(
        **results,
        auc_entry=auc_entry,
        auc_duration=auc_dur,
        shap_sv=sv,
        shap_sample=shap_sample,
        shap_df=shap_df,
    )
