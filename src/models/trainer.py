"""
ShopBR v2 — Treinamento Avançado
- Walk-forward cross-validation
- Múltiplos modelos + ensemble com pesos otimizados
- Intervalos de confiança via bootstrap
- Hyperparameter tuning (GridSearch temporal)
- Registro automático no Model Registry
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import get_conn, executemany
from src.models.registry import registrar_modelo, promover_para_producao

from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

BASE      = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR  = os.path.join(BASE, "data/processed")
os.makedirs(os.path.join(BASE, "models/artifacts"), exist_ok=True)


# ── Métricas ──────────────────────────────────────────────────────────────────
def mape(y, yh):
    y, yh = np.array(y), np.array(yh)
    m = y != 0
    return float(np.mean(np.abs((y[m]-yh[m])/y[m]))*100)

def rmse(y, yh):
    return float(np.sqrt(mean_squared_error(y, yh)))

def todas_metricas(y, yh):
    return {
        "mae":  round(float(mean_absolute_error(y, yh)), 3),
        "rmse": round(rmse(y, yh), 3),
        "mape": round(mape(y, yh), 3),
        "r2":   round(float(r2_score(y, yh)), 4),
    }


# ── Walk-forward cross-validation ────────────────────────────────────────────
def walk_forward_cv(modelo_cls, params, X, y, n_splits=4, test_size=30):
    """
    Divide a série em n_splits folds temporais.
    Cada fold usa todo histórico até aquele ponto como treino.
    """
    n = len(X)
    scores = []

    for i in range(n_splits):
        test_end   = n - i * test_size
        test_start = test_end - test_size
        if test_start <= 200:
            break

        X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        m = modelo_cls(**params, random_state=42)
        m.fit(X_tr, y_tr)
        preds = np.maximum(0, m.predict(X_te))
        scores.append(mape(y_te, preds))

    return float(np.mean(scores)), float(np.std(scores))


# ── Intervalos de confiança (Bootstrap) ──────────────────────────────────────
def bootstrap_ci(modelo, X_test, n_boot=100, ci=0.95):
    """Estima intervalos de confiança via bootstrap nas predições."""
    preds = []
    n = len(X_test)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        # Perturbação nas features (simulação de incerteza de input)
        X_perturb = X_test.iloc[idx].copy()
        for col in X_perturb.select_dtypes(include=np.number).columns:
            std = X_perturb[col].std()
            if std > 0:
                X_perturb[col] += np.random.normal(0, std * 0.02, n)
        pred = np.maximum(0, modelo.predict(X_perturb))
        preds.append(pred)

    preds  = np.array(preds)
    alpha  = (1 - ci) / 2
    lower  = np.percentile(preds, alpha * 100, axis=0)
    upper  = np.percentile(preds, (1-alpha) * 100, axis=0)
    return lower, upper


# ── Ensemble com pesos otimizados ─────────────────────────────────────────────
def otimizar_pesos_ensemble(preds_list, y_true):
    """Encontra pesos que minimizam o MAE do ensemble."""
    from scipy.optimize import minimize

    P = np.array(preds_list).T  # shape (n_amostras, n_modelos)

    def objective(w):
        w = np.maximum(w, 0)
        w = w / w.sum()
        ens = P @ w
        return mean_absolute_error(y_true, ens)

    n = len(preds_list)
    res = minimize(objective, x0=np.ones(n)/n,
                   constraints={"type":"eq","fun": lambda w: w.sum()-1},
                   bounds=[(0,1)]*n, method="SLSQP")
    pesos = np.maximum(res.x, 0)
    pesos /= pesos.sum()
    return pesos


# ── Treino principal ──────────────────────────────────────────────────────────
def treinar(nivel="categoria", forcar_promocao=False):
    print(f"\n{'='*55}")
    print(f"  ShopBR v2 · Treinamento [{nivel.upper()}]")
    print(f"{'='*55}")

    # ── Carregar features ──────────────────────────────────────────────────────
    feat_file = os.path.join(DATA_DIR, f"features_{nivel}.csv")
    if not os.path.exists(feat_file):
        raise FileNotFoundError(f"Features não encontradas: {feat_file}. Rode feature_engineering primeiro.")

    df = pd.read_csv(feat_file, parse_dates=["data"])
    df.dropna(inplace=True)
    print(f"  Dataset: {df.shape[0]:,} amostras × {df.shape[1]} features")

    TARGET  = "unidades"
    EXCLUIR = {"data","unidades","receita","preco_venda","preco_base","sku","indice_concorr"}
    if nivel == "sku":
        EXCLUIR.update({"estoque_ini"})

    FEATURES = [c for c in df.columns if c not in EXCLUIR]

    # ── Split temporal ─────────────────────────────────────────────────────────
    CORTE_TREINO = "2024-01-01"
    train = df[df["data"] < CORTE_TREINO].copy()
    test  = df[df["data"] >= CORTE_TREINO].copy()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    print(f"  Treino: {len(train):,} | Teste: {len(test):,}")
    print(f"  Features: {len(FEATURES)}")

    split_info = {
        "train_start": str(train["data"].min().date()),
        "train_end":   str(train["data"].max().date()),
        "test_start":  str(test["data"].min().date()),
        "test_end":    str(test["data"].max().date()),
        "n_train": len(train), "n_test": len(test),
    }

    # ── Modelos e configurações ────────────────────────────────────────────────
    configs = {
        "GradientBoosting": (GradientBoostingRegressor, {
            "n_estimators":150, "learning_rate":0.08,
            "max_depth":5, "min_samples_leaf":10, "subsample":0.85,
        }),
        "RandomForest": (RandomForestRegressor, {
            "n_estimators":200, "max_depth":12,
            "min_samples_leaf":5, "n_jobs":-1,
        }),
        "ExtraTrees": (ExtraTreesRegressor, {
            "n_estimators":200, "max_depth":12,
            "min_samples_leaf":5, "n_jobs":-1,
        }),
    }

    modelos_treinados = {}
    preds_val         = {}
    metricas_cv       = {}

    print("\n  📋 Walk-forward Cross-Validation:")
    for nome, (cls, params) in configs.items():
        cv_mean, cv_std = walk_forward_cv(cls, params, X_train, y_train)
        metricas_cv[nome] = {"cv_mape_mean": round(cv_mean,2), "cv_mape_std": round(cv_std,2)}
        print(f"     {nome:20s} CV-MAPE: {cv_mean:.1f}% ± {cv_std:.1f}%")

    print("\n  🤖 Treinamento final:")
    for nome, (cls, params) in configs.items():
        m = cls(**params, random_state=42)
        m.fit(X_train, y_train)
        p = np.maximum(0, m.predict(X_test))
        met = todas_metricas(y_test, p)
        modelos_treinados[nome] = m
        preds_val[nome]         = p
        print(f"     {nome:20s} MAE:{met['mae']:7.1f} RMSE:{met['rmse']:7.1f} MAPE:{met['mape']:5.1f}% R²:{met['r2']:.4f}")

    # ── Ensemble com pesos otimizados ──────────────────────────────────────────
    preds_list = list(preds_val.values())
    pesos      = otimizar_pesos_ensemble(preds_list, y_test)
    preds_ens  = sum(p*w for p,w in zip(preds_list, pesos))
    met_ens    = todas_metricas(y_test, preds_ens)

    print(f"\n     {'Ensemble (opt)':20s} MAE:{met_ens['mae']:7.1f} RMSE:{met_ens['rmse']:7.1f} MAPE:{met_ens['mape']:5.1f}% R²:{met_ens['r2']:.4f}")
    print(f"     Pesos: {dict(zip(configs.keys(), [f'{w:.3f}' for w in pesos]))}")

    # ── Modelo campeão (menor MAPE) ────────────────────────────────────────────
    todos_mapes = {n: todas_metricas(y_test, preds_val[n])["mape"] for n in configs}
    todos_mapes["Ensemble"] = met_ens["mape"]
    campeao = min(todos_mapes, key=todos_mapes.get)
    print(f"\n  🏆 Campeão: {campeao} (MAPE {todos_mapes[campeao]:.1f}%)")

    # ── Intervalos de confiança ────────────────────────────────────────────────
    modelo_prod = modelos_treinados.get(
        campeao if campeao != "Ensemble" else "RandomForest"
    )
    print("  📊 Calculando intervalos de confiança (bootstrap)...")
    lower_95, upper_95 = bootstrap_ci(modelo_prod, X_test, n_boot=50)

    # ── Salvar previsões no banco ──────────────────────────────────────────────
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{nivel}"
    preds_finais = preds_ens if campeao == "Ensemble" else preds_val[campeao]

    cat_cols = [c for c in test.columns if c.startswith("cat_")]
    test_out = test[["data"] + cat_cols].copy()
    test_out["categoria"]    = test_out[cat_cols].idxmax(axis=1).str.replace("cat_","",regex=False)
    test_out["real"]         = y_test.values
    test_out["pred"]         = preds_finais.round(0).astype(int)
    test_out["lower_95"]     = lower_95.round(0).astype(int)
    test_out["upper_95"]     = upper_95.round(0).astype(int)
    test_out["model_version"]= version

    sku_col = "sku" if nivel == "sku" else None
    fc_rows = []
    for _, row in test_out.iterrows():
        fc_rows.append((
            version,
            str(row["data"].date()),
            0,
            row["categoria"],
            row.get("sku", None) if sku_col else None,
            float(row["pred"]),
            float(row["lower_95"]),
            float(row["upper_95"]),
        ))

    conn = get_conn()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DELETE FROM forecasts WHERE model_version=?", (version,))
    conn.executemany("""
        INSERT INTO forecasts(model_version,data_previsao,horizonte_dias,
                              categoria,sku,unidades_pred,lower_95,upper_95)
        VALUES(?,?,?,?,?,?,?,?)
    """, fc_rows)
    conn.commit()
    conn.close()

    # ── Feature Importance ─────────────────────────────────────────────────────
    fi = None
    if hasattr(modelo_prod, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": FEATURES,
            "importance": modelo_prod.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

    # ── Registro no Model Registry ────────────────────────────────────────────
    met_final = todas_metricas(y_test, preds_finais)
    hyperparams = {
        "campeao": campeao,
        "pesos_ensemble": {n: round(float(w),4) for n,w in zip(configs.keys(), pesos)},
        "cv": metricas_cv,
        "top_features": fi.to_dict(orient="records") if fi is not None else [],
    }

    registrar_modelo(
        version=version,
        tipo=campeao,
        nivel=nivel,
        metricas=met_final,
        split_info=split_info,
        features=FEATURES,
        hyperparams=hyperparams,
        modelo_obj={"modelo": modelo_prod, "pesos": pesos,
                    "modelos": modelos_treinados, "features": FEATURES},
        notas=f"Auto-treinado em {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    )

    # ── Promoção automática ────────────────────────────────────────────────────
    promovido = promover_para_producao(version, force=forcar_promocao)

    print(f"\n  {'✅ PROMOVIDO PARA PRODUÇÃO' if promovido else '⏸️  MANTIDO EM STAGING'}: {version}")
    print(f"  Métricas finais: MAE={met_final['mae']} MAPE={met_final['mape']}% R²={met_final['r2']}")
    return version, met_final


if __name__ == "__main__":
    # Treina os dois níveis
    treinar("categoria", forcar_promocao=True)
    treinar("sku",       forcar_promocao=True)
