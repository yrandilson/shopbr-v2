"""
ShopBR v2 — API Flask (produção-grade)
Endpoints REST para dashboard + integração com sistemas externos (ERP, BI).
"""

import sys, os, json
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_conn, query_df
from src.models.registry import listar_modelos, get_production_model, rollback as do_rollback
from src.monitoring.drift_monitor import relatorio_saude, monitorar_features, monitorar_previsoes

app = Flask(__name__, template_folder="templates", static_folder="static")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rows_to_list(rows):
    return [dict(r) for r in rows]

def _ok(data):
    return jsonify({"status": "ok", "data": data})

def _erro(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Overview / KPIs ───────────────────────────────────────────────────────────
@app.route("/api/overview")
def api_overview():
    conn = get_conn()

    # Modelo em produção
    mod = conn.execute("""
        SELECT version, modelo_tipo, mape, r2, mae, rmse, promovido_em, treinado_em
        FROM model_registry WHERE status='production' AND nivel='categoria'
        ORDER BY promovido_em DESC LIMIT 1
    """).fetchone()

    # Totais de vendas
    totais = conn.execute("""
        SELECT SUM(unidades) as total_unidades, SUM(receita) as total_receita,
               COUNT(DISTINCT sku) as n_skus, COUNT(DISTINCT categoria) as n_cats,
               MIN(data) as data_inicio, MAX(data) as data_fim
        FROM sales
    """).fetchone()

    # Previsões período de teste
    fc = conn.execute("""
        SELECT SUM(unidades_pred) as total_pred,
               COUNT(*) as n_preds
        FROM forecasts
        WHERE model_version = (
            SELECT version FROM model_registry WHERE status='production' AND nivel='categoria' LIMIT 1
        )
    """).fetchone()

    # Real do período de teste
    real = conn.execute("""
        SELECT SUM(unidades) as total_real
        FROM sales WHERE data >= '2024-01-01'
    """).fetchone()

    conn.close()

    return _ok({
        "modelo": dict(mod) if mod else None,
        "vendas": dict(totais),
        "previsoes": {
            "total_pred": int(fc["total_pred"] or 0),
            "total_real": int(real["total_real"] or 0),
            "n_preds":    int(fc["n_preds"] or 0),
        }
    })


# ── Série temporal: real vs previsto ─────────────────────────────────────────
@app.route("/api/serie_temporal")
def api_serie_temporal():
    cat    = request.args.get("categoria", "Todas")
    nivel  = request.args.get("nivel", "categoria")
    agrup  = request.args.get("agrupamento", "semana")

    # Previsões
    where_cat = f"AND f.categoria = '{cat}'" if cat != "Todas" else ""
    fc_df = query_df(f"""
        SELECT f.data_previsao as data, SUM(f.unidades_pred) as pred,
               SUM(f.lower_95) as lower_95, SUM(f.upper_95) as upper_95,
               SUM(s.unidades) as real
        FROM forecasts f
        JOIN sales s ON f.data_previsao = s.data AND f.categoria = s.categoria
        WHERE f.model_version = (
            SELECT version FROM model_registry
            WHERE status='production' AND nivel='categoria'
            ORDER BY promovido_em DESC LIMIT 1
        ) {where_cat}
        GROUP BY f.data_previsao
        ORDER BY f.data_previsao
    """)

    if fc_df.empty:
        return _ok({"labels":[],"real":[],"pred":[],"lower":[],"upper":[]})

    fc_df["data"] = pd.to_datetime(fc_df["data"])

    if agrup == "semana":
        fc_df["periodo"] = fc_df["data"].dt.to_period("W").dt.start_time
    else:
        fc_df["periodo"] = fc_df["data"].dt.to_period("M").dt.start_time

    grp = fc_df.groupby("periodo").sum(numeric_only=True).reset_index()
    return _ok({
        "labels":   grp["periodo"].dt.strftime("%d/%m/%Y").tolist(),
        "real":     grp["real"].round(0).astype(int).tolist(),
        "pred":     grp["pred"].round(0).astype(int).tolist(),
        "lower_95": grp["lower_95"].round(0).astype(int).tolist(),
        "upper_95": grp["upper_95"].round(0).astype(int).tolist(),
    })


# ── Histórico completo ────────────────────────────────────────────────────────
@app.route("/api/historico")
def api_historico():
    import pandas as pd
    cat  = request.args.get("categoria", "Todas")
    where = f"WHERE categoria='{cat}'" if cat != "Todas" else ""

    df = query_df(f"""
        SELECT data, SUM(unidades) as unidades, SUM(receita)/1000 as receita_k
        FROM sales {where}
        GROUP BY data ORDER BY data
    """)

    df["data"]    = pd.to_datetime(df["data"])
    df["semana"]  = df["data"].dt.to_period("W").dt.start_time
    grp = df.groupby("semana").sum(numeric_only=True).reset_index()

    return _ok({
        "labels":   grp["semana"].dt.strftime("%d/%m/%Y").tolist(),
        "unidades": grp["unidades"].tolist(),
        "receita_k":grp["receita_k"].round(1).tolist(),
    })


# ── Por categoria ─────────────────────────────────────────────────────────────
@app.route("/api/por_categoria")
def api_por_categoria():
    df = query_df("""
        SELECT f.categoria,
               SUM(f.unidades_pred) as pred,
               SUM(s.unidades)      as real,
               SUM(s.receita)       as receita
        FROM forecasts f
        JOIN sales s ON f.data_previsao = s.data AND f.categoria = s.categoria
        WHERE f.model_version = (
            SELECT version FROM model_registry WHERE status='production' AND nivel='categoria' LIMIT 1
        )
        GROUP BY f.categoria
    """)
    if df.empty:
        return _ok([])

    df["erro_pct"]  = ((df["pred"] - df["real"]) / df["real"] * 100).round(1)
    df["pred"]      = df["pred"].round(0).astype(int)
    df["real"]      = df["real"].round(0).astype(int)
    df["receita"]   = df["receita"].round(0).astype(int)
    return _ok(df.to_dict(orient="records"))


# ── Feature importance ────────────────────────────────────────────────────────
@app.route("/api/feature_importance")
def api_feature_importance():
    conn = get_conn()
    row = conn.execute("""
        SELECT hyperparams FROM model_registry
        WHERE status='production' AND nivel='categoria' LIMIT 1
    """).fetchone()
    conn.close()

    if not row:
        return _ok([])

    hp   = json.loads(row["hyperparams"])
    top  = hp.get("top_features", [])[:15]

    label_map = {
        "lag_1d":"Lag 1 dia","lag_7d":"Lag 7 dias","lag_14d":"Lag 14 dias",
        "lag_21d":"Lag 21 dias","lag_28d":"Lag 28 dias","lag_35d":"Lag 35 dias",
        "roll_mean_7d":"Média 7d","roll_std_7d":"Desvio 7d","roll_min_7d":"Mín 7d",
        "roll_min_14d":"Mín 14d","roll_mean_14d":"Média 14d","roll_med_7d":"Mediana 7d",
        "aceleracao_7_28":"Aceleração 7/28d","lag_concorr_7d":"Concorrência 7d",
        "promo_lag_1d":"Promoção ontem","promo_lag_7d":"Promoções 7d",
        "semana_ano":"Semana do Ano","dia_mes":"Dia do Mês",
        "dia_semana":"Dia Semana","diff_28d":"Δ 28 dias",
        "mes":"Mês","mes_sin":"Mês (sin)","trend_idx":"Tendência",
        "roll_receita_7d":"Receita 7d","is_feriado":"Feriado Comercial",
    }
    return _ok([
        {"feature": label_map.get(r["feature"], r["feature"]),
         "importance": round(r["importance"]*100, 2)}
        for r in top
    ])


# ── Resíduos ──────────────────────────────────────────────────────────────────
@app.route("/api/residuos")
def api_residuos():
    import numpy as np
    df = query_df("""
        SELECT f.unidades_pred as pred, SUM(s.unidades) as real
        FROM forecasts f
        JOIN sales s ON f.data_previsao = s.data AND f.categoria = s.categoria
        WHERE f.model_version = (
            SELECT version FROM model_registry WHERE status='production' AND nivel='categoria' LIMIT 1
        )
        GROUP BY f.data_previsao, f.categoria
    """)

    if df.empty:
        return _ok({"bins":[],"counts":[]})

    residuos = (df["real"] - df["pred"]).values
    hist, edges = np.histogram(residuos, bins=35)
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]

    return _ok({
        "bins":   [round(c,1) for c in centers],
        "counts": hist.tolist(),
        "mean":   round(float(residuos.mean()), 2),
        "std":    round(float(residuos.std()), 2),
    })


# ── Model Registry ────────────────────────────────────────────────────────────
@app.route("/api/modelos")
def api_modelos():
    df = listar_modelos()
    return _ok(df.to_dict(orient="records"))


@app.route("/api/modelos/rollback", methods=["POST"])
def api_rollback():
    nivel = request.json.get("nivel", "categoria")
    ok = do_rollback(nivel)
    if ok:
        return _ok({"message": f"Rollback executado para [{nivel}]"})
    return _erro("Nenhum modelo archived disponível")


# ── Drift Monitoring ──────────────────────────────────────────────────────────
@app.route("/api/monitoring/saude")
def api_saude():
    rel = relatorio_saude()
    return _ok(rel)


@app.route("/api/monitoring/drift_log")
def api_drift_log():
    df = query_df("""
        SELECT data_check, feature, metrica, valor, threshold, alerta, criado_em
        FROM drift_log
        ORDER BY criado_em DESC
        LIMIT 100
    """)
    return _ok(df.to_dict(orient="records"))


@app.route("/api/monitoring/retraining_log")
def api_retraining_log():
    df = query_df("""
        SELECT * FROM retraining_log
        ORDER BY executado_em DESC LIMIT 20
    """)
    return _ok(df.to_dict(orient="records"))


# ── Retreinamento manual ──────────────────────────────────────────────────────
@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    import threading
    nivel  = request.json.get("nivel", "categoria")
    forcar = request.json.get("forcar", False)

    def run():
        from src.scheduler.auto_retrain import executar_retreinamento
        executar_retreinamento(trigger="manual", nivel=nivel, forcar=forcar)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return _ok({"message": f"Retreinamento [{nivel}] iniciado em background"})


# ── SKU ranking ───────────────────────────────────────────────────────────────
@app.route("/api/sku/ranking")
def api_sku_ranking():
    cat   = request.args.get("categoria", "Todas")
    where = f"WHERE s.categoria='{cat}'" if cat != "Todas" else ""
    df = query_df(f"""
        SELECT s.sku, p.nome, s.categoria,
               SUM(s.unidades) as total_unidades,
               SUM(s.receita)  as total_receita,
               AVG(s.promocao) * 100 as pct_promocao,
               AVG(s.indice_concorr) as indice_concorr_medio
        FROM sales s
        JOIN products p ON s.sku = p.sku
        {where}
        GROUP BY s.sku
        ORDER BY total_receita DESC
        LIMIT 20
    """)
    df["total_receita"]   = df["total_receita"].round(0).astype(int)
    df["pct_promocao"]    = df["pct_promocao"].round(1)
    df["indice_concorr_medio"] = df["indice_concorr_medio"].round(3)
    return _ok(df.to_dict(orient="records"))


# ── Categorias disponíveis ────────────────────────────────────────────────────
@app.route("/api/categorias")
def api_categorias():
    conn = get_conn()
    cats = [r[0] for r in conn.execute("SELECT DISTINCT categoria FROM sales ORDER BY categoria").fetchall()]
    conn.close()
    return _ok(cats)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
