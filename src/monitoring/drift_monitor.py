"""
ShopBR v2 — Monitoramento de Drift
Detecta degradação do modelo em produção via:
- PSI (Population Stability Index) nas features
- KS Test (distribuição de previsões vs real)
- Rolling MAPE (janela deslizante de 7 dias)
- Alertas automáticos com log no banco
"""

import pandas as pd
import numpy as np
import sys, os, json
from datetime import datetime, timedelta
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import get_conn, query_df
from src.models.registry import get_production_model

# ── Thresholds ────────────────────────────────────────────────────────────────
PSI_THRESHOLD       = 0.20   # > 0.20 = drift severo (requer retreinamento)
PSI_WARNING         = 0.10   # 0.10-0.20 = atenção
KS_PVALUE_THRESHOLD = 0.05   # p < 0.05 = distribuições diferentes
MAPE_THRESHOLD      = 30.0   # MAPE > 30% rolling = alerta


def calcular_psi(referencia, atual, bins=10):
    """
    PSI: mede quanto a distribuição de uma feature mudou.
    PSI < 0.10 → estável
    PSI 0.10–0.20 → atenção
    PSI > 0.20 → drift severo
    """
    ref_clip = np.clip(referencia, np.percentile(referencia, 1), np.percentile(referencia, 99))
    boundaries = np.percentile(ref_clip, np.linspace(0, 100, bins+1))
    boundaries = np.unique(boundaries)
    if len(boundaries) < 3:
        return 0.0

    ref_counts = np.histogram(referencia, bins=boundaries)[0] + 1e-6
    atu_counts = np.histogram(atual,      bins=boundaries)[0] + 1e-6

    ref_pct = ref_counts / ref_counts.sum()
    atu_pct = atu_counts / atu_counts.sum()

    psi = np.sum((atu_pct - ref_pct) * np.log(atu_pct / ref_pct))
    return float(psi)


def monitorar_features(nivel="categoria", janela_dias=30):
    """
    Compara distribuição das features do treino (referência)
    vs últimas `janela_dias` dias (atual).
    """
    _, meta = get_production_model(nivel)
    if meta is None:
        print("  ❌ Nenhum modelo em produção")
        return []

    feat_file = os.path.join(
        os.path.dirname(__file__), f"../../data/processed/features_{nivel}.csv"
    )
    df = pd.read_csv(feat_file, parse_dates=["data"])
    df.dropna(inplace=True)

    FEATURES = [f for f in meta["features"] if f in df.columns]
    num_features = df[FEATURES].select_dtypes(include=np.number).columns.tolist()

    data_corte = df["data"].max() - timedelta(days=janela_dias)
    ref  = df[df["data"] < meta["train_end"]][num_features]
    atu  = df[df["data"] >= data_corte][num_features]

    if len(atu) < 10:
        print("  ⚠️  Dados recentes insuficientes para monitoramento")
        return []

    alertas = []
    hoje    = datetime.now().strftime("%Y-%m-%d")

    conn = get_conn()
    for feat in num_features[:20]:  # top 20 features
        try:
            psi_val = calcular_psi(ref[feat].dropna().values, atu[feat].dropna().values)
            alerta  = 1 if psi_val > PSI_THRESHOLD else 0

            conn.execute("""
                INSERT INTO drift_log(data_check, model_version, feature,
                                      metrica, valor, threshold, alerta)
                VALUES(?,?,?,?,?,?,?)
            """, (hoje, meta["version"], feat, "psi",
                  round(psi_val, 4), PSI_THRESHOLD, alerta))

            if alerta:
                alertas.append({"feature": feat, "psi": round(psi_val, 4)})
        except Exception:
            pass

    conn.commit()
    conn.close()

    return alertas


def monitorar_previsoes(nivel="categoria", janela_dias=14):
    """
    Compara distribuição de previsões vs real no período recente.
    Usa teste KS e rolling MAPE.
    """
    conn = get_conn()
    hoje = datetime.now().strftime("%Y-%m-%d")

    # Busca previsões e real
    rows = conn.execute("""
        SELECT f.data_previsao as data, f.unidades_pred as pred,
               SUM(s.unidades) as real
        FROM forecasts f
        JOIN sales s ON f.data_previsao = s.data AND f.categoria = s.categoria
        WHERE f.model_version = (
            SELECT version FROM model_registry
            WHERE status='production' AND nivel=?
            ORDER BY promovido_em DESC LIMIT 1
        )
        GROUP BY f.data_previsao, f.categoria
        ORDER BY f.data_previsao DESC
        LIMIT ?
    """, (nivel, janela_dias * 10)).fetchall()
    conn.close()

    if not rows or len(rows) < 5:
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    df.dropna(inplace=True)

    # KS Test
    ks_stat, ks_p = stats.ks_2samp(df["pred"].values, df["real"].values)

    # Rolling MAPE
    mask = df["real"] > 0
    mape_val = float(np.mean(np.abs((df.loc[mask,"real"] - df.loc[mask,"pred"]) / df.loc[mask,"real"])) * 100)

    alerta_ks   = 1 if ks_p < KS_PVALUE_THRESHOLD else 0
    alerta_mape = 1 if mape_val > MAPE_THRESHOLD else 0

    conn = get_conn()
    version = conn.execute(
        "SELECT version FROM model_registry WHERE status='production' AND nivel=? LIMIT 1",
        (nivel,)
    ).fetchone()
    version = version["version"] if version else "unknown"

    conn.execute("""
        INSERT INTO drift_log(data_check, model_version, feature, metrica, valor, threshold, alerta)
        VALUES(?,?,?,?,?,?,?)
    """, (hoje, version, "predicoes", "ks_stat", round(float(ks_stat),4), KS_PVALUE_THRESHOLD, alerta_ks))

    conn.execute("""
        INSERT INTO drift_log(data_check, model_version, feature, metrica, valor, threshold, alerta)
        VALUES(?,?,?,?,?,?,?)
    """, (hoje, version, "rolling_mape", "mape_rolling", round(mape_val,2), MAPE_THRESHOLD, alerta_mape))

    conn.commit()
    conn.close()

    return {
        "ks_stat": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_p), 4),
        "rolling_mape": round(mape_val, 2),
        "alerta_ks": alerta_ks,
        "alerta_mape": alerta_mape,
    }


def precisa_retreinar(nivel="categoria"):
    """
    Verifica se o modelo precisa de retreinamento baseado nos logs de drift.
    Retorna (bool, motivo).
    """
    conn = get_conn()
    alertas = conn.execute("""
        SELECT COUNT(*) as n, metrica
        FROM drift_log
        WHERE alerta=1
          AND data_check >= date('now', '-7 days')
          AND model_version = (
              SELECT version FROM model_registry
              WHERE status='production' AND nivel=?
              ORDER BY promovido_em DESC LIMIT 1
          )
        GROUP BY metrica
    """, (nivel,)).fetchall()
    conn.close()

    for a in alertas:
        n, metrica = a["n"], a["metrica"]
        if metrica == "psi" and n >= 3:
            return True, f"PSI drift em {n} features nos últimos 7 dias"
        if metrica == "mape_rolling" and n >= 2:
            return True, f"MAPE rolling acima de {MAPE_THRESHOLD}% por {n} dias"

    return False, "Modelo estável"


def relatorio_saude():
    """Gera relatório completo de saúde dos modelos."""
    resultado = {}
    for nivel in ["categoria", "sku"]:
        _, meta = get_production_model(nivel)
        if not meta:
            resultado[nivel] = {"status": "sem_modelo"}
            continue

        alertas_feat  = monitorar_features(nivel)
        alertas_pred  = monitorar_previsoes(nivel)
        retratar, mot = precisa_retreinar(nivel)

        resultado[nivel] = {
            "version":        meta["version"],
            "mape_treino":    meta["mape"],
            "alertas_features": len(alertas_feat),
            "features_drift": alertas_feat[:5],
            "rolling_mape":   alertas_pred.get("rolling_mape"),
            "ks_stat":        alertas_pred.get("ks_stat"),
            "requer_retreino": retratar,
            "motivo":         mot,
        }

    return resultado


if __name__ == "__main__":
    print("🔍 Executando monitoramento de drift...")
    rel = relatorio_saude()
    for nivel, info in rel.items():
        print(f"\n  [{nivel.upper()}]")
        for k, v in info.items():
            print(f"    {k}: {v}")
