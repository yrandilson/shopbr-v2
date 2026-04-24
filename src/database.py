"""
ShopBR v2 — Database Layer
SQLite com schema de produção (simula PostgreSQL).
Tabelas: products, sales, forecasts, model_registry, drift_log, retraining_log
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/shopbr.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c = conn.cursor()

    # ── Produtos (SKU level) ──────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        sku         TEXT PRIMARY KEY,
        nome        TEXT NOT NULL,
        categoria   TEXT NOT NULL,
        preco_base  REAL NOT NULL,
        ativo       INTEGER DEFAULT 1,
        criado_em   TEXT DEFAULT (datetime('now'))
    )""")

    # ── Vendas brutas ─────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        data            TEXT NOT NULL,
        sku             TEXT NOT NULL,
        categoria       TEXT NOT NULL,
        unidades        INTEGER NOT NULL,
        preco_venda     REAL NOT NULL,
        receita         REAL NOT NULL,
        promocao        INTEGER DEFAULT 0,
        estoque_ini     INTEGER DEFAULT 0,
        indice_concorr  REAL DEFAULT 1.0,
        FOREIGN KEY (sku) REFERENCES products(sku)
    )""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_sales_data ON sales(data)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sales_cat  ON sales(categoria)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sales_sku  ON sales(sku)")

    # ── Model Registry ────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        version         TEXT NOT NULL UNIQUE,
        modelo_tipo     TEXT NOT NULL,
        nivel           TEXT NOT NULL,        -- 'categoria' ou 'sku'
        status          TEXT DEFAULT 'staging', -- staging | production | archived
        mae             REAL,
        rmse            REAL,
        mape            REAL,
        r2              REAL,
        train_start     TEXT,
        train_end       TEXT,
        test_start      TEXT,
        test_end        TEXT,
        n_train         INTEGER,
        n_test          INTEGER,
        features        TEXT,                 -- JSON list
        hyperparams     TEXT,                 -- JSON dict
        artifact_path   TEXT,
        treinado_em     TEXT DEFAULT (datetime('now')),
        promovido_em    TEXT,
        notas           TEXT
    )""")

    # ── Previsões armazenadas ─────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version   TEXT NOT NULL,
        data_previsao   TEXT NOT NULL,
        horizonte_dias  INTEGER NOT NULL,
        categoria       TEXT NOT NULL,
        sku             TEXT,
        unidades_pred   REAL NOT NULL,
        lower_95        REAL,
        upper_95        REAL,
        gerado_em       TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (model_version) REFERENCES model_registry(version)
    )""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_fc_data ON forecasts(data_previsao)")

    # ── Drift Log ─────────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS drift_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        data_check      TEXT NOT NULL,
        model_version   TEXT NOT NULL,
        feature         TEXT NOT NULL,
        metrica         TEXT NOT NULL,   -- 'psi' | 'ks_stat' | 'mape_rolling'
        valor           REAL NOT NULL,
        threshold       REAL NOT NULL,
        alerta          INTEGER DEFAULT 0,
        detalhes        TEXT,
        criado_em       TEXT DEFAULT (datetime('now'))
    )""")

    # ── Retraining Log ────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS retraining_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        trigger         TEXT NOT NULL,   -- 'scheduled' | 'drift_alert' | 'manual'
        status          TEXT NOT NULL,   -- 'success' | 'failed' | 'skipped'
        version_anterior TEXT,
        version_nova    TEXT,
        mape_anterior   REAL,
        mape_novo       REAL,
        melhoria_pct    REAL,
        duracao_seg     REAL,
        detalhes        TEXT,
        executado_em    TEXT DEFAULT (datetime('now'))
    )""")

    conn.commit()
    conn.close()
    print(f"✅ DB inicializado: {DB_PATH}")


def query_df(sql, params=()):
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def execute(sql, params=()):
    conn = get_conn()
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def executemany(sql, rows):
    conn = get_conn()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
