"""
ShopBR v2 — Model Registry
Controle de versões de modelos: staging → production → archived
Suporta rollback automático.
"""

import json
import os
import joblib
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import get_conn, query_df

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../../models/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar_modelo(version, tipo, nivel, metricas, split_info,
                     features, hyperparams, modelo_obj=None, notas=""):
    """Registra um novo modelo no registry com status 'staging'."""
    artifact_path = os.path.join(ARTIFACTS_DIR, f"{version}.pkl")
    if modelo_obj is not None:
        joblib.dump(modelo_obj, artifact_path)

    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO model_registry
        (version, modelo_tipo, nivel, status, mae, rmse, mape, r2,
         train_start, train_end, test_start, test_end, n_train, n_test,
         features, hyperparams, artifact_path, notas)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        version, tipo, nivel, "staging",
        metricas["mae"], metricas["rmse"], metricas["mape"], metricas["r2"],
        split_info["train_start"], split_info["train_end"],
        split_info["test_start"],  split_info["test_end"],
        split_info["n_train"],     split_info["n_test"],
        json.dumps(features, ensure_ascii=False),
        json.dumps(hyperparams, ensure_ascii=False),
        artifact_path, notas
    ))
    conn.commit()
    conn.close()
    print(f"  📋 Registrado: {version} → staging")


def promover_para_producao(version, force=False):
    """
    Promove modelo para produção.
    Se force=False, só promove se MAPE for melhor que o atual em produção.
    Arquiva o modelo anterior.
    """
    conn = get_conn()

    # Busca candidato
    cand = conn.execute(
        "SELECT * FROM model_registry WHERE version=?", (version,)
    ).fetchone()

    if not cand:
        conn.close()
        raise ValueError(f"Versão {version} não encontrada")

    # Busca modelo em produção atual
    atual = conn.execute(
        "SELECT * FROM model_registry WHERE status='production' ORDER BY promovido_em DESC LIMIT 1"
    ).fetchone()

    if atual and not force:
        if cand["mape"] >= atual["mape"]:
            conn.close()
            print(f"  ⚠️  Promoção recusada: MAPE {cand['mape']:.2f}% ≥ atual {atual['mape']:.2f}%")
            return False

    # Arquiva o atual
    if atual:
        conn.execute(
            "UPDATE model_registry SET status='archived' WHERE version=?",
            (atual["version"],)
        )
        print(f"  📦 Arquivado: {atual['version']}")

    # Promove o novo
    conn.execute("""
        UPDATE model_registry
        SET status='production', promovido_em=?
        WHERE version=?
    """, (_now(), version))

    conn.commit()
    conn.close()
    print(f"  🚀 Promovido para produção: {version}")
    return True


def rollback(nivel="categoria"):
    """Reverte para o modelo em produção anterior (archived mais recente)."""
    conn = get_conn()

    prod = conn.execute(
        "SELECT * FROM model_registry WHERE status='production' AND nivel=? LIMIT 1",
        (nivel,)
    ).fetchone()

    anterior = conn.execute("""
        SELECT * FROM model_registry
        WHERE status='archived' AND nivel=?
        ORDER BY promovido_em DESC LIMIT 1
    """, (nivel,)).fetchone()

    if not anterior:
        conn.close()
        print("  ❌ Nenhum modelo archived disponível para rollback")
        return False

    if prod:
        conn.execute(
            "UPDATE model_registry SET status='archived' WHERE version=?",
            (prod["version"],)
        )

    conn.execute(
        "UPDATE model_registry SET status='production', promovido_em=? WHERE version=?",
        (_now(), anterior["version"])
    )

    conn.commit()
    conn.close()
    print(f"  ↩️  Rollback: {prod['version'] if prod else '?'} → {anterior['version']}")
    return True


def get_production_model(nivel="categoria"):
    """Retorna o modelo em produção (objeto + metadados)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM model_registry WHERE status='production' AND nivel=? ORDER BY promovido_em DESC LIMIT 1",
        (nivel,)
    ).fetchone()
    conn.close()

    if not row:
        return None, None

    meta = dict(row)
    meta["features"]    = json.loads(meta["features"])
    meta["hyperparams"] = json.loads(meta["hyperparams"])

    modelo = joblib.load(meta["artifact_path"])
    return modelo, meta


def listar_modelos():
    return query_df("SELECT version, modelo_tipo, nivel, status, mape, r2, treinado_em, promovido_em FROM model_registry ORDER BY treinado_em DESC")


if __name__ == "__main__":
    print(listar_modelos())
