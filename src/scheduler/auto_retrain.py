"""
ShopBR v2 — Scheduler de Retreinamento Automático
Simula o Airflow/cron de produção:
- Job semanal (toda segunda-feira)
- Job de emergência (quando drift detectado)
- Rollback automático se novo modelo piorar
"""

import sys, os, json, time, threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_conn
from src.monitoring.drift_monitor import precisa_retreinar, relatorio_saude
from src.models.registry import get_production_model, rollback


def _log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def _registrar_log_retreino(trigger, status, version_ant, version_novo,
                             mape_ant, mape_novo, duracao, detalhes=""):
    melhoria = None
    if mape_ant and mape_novo:
        melhoria = round((mape_ant - mape_novo) / mape_ant * 100, 2)

    conn = get_conn()
    conn.execute("""
        INSERT INTO retraining_log
        (trigger, status, version_anterior, version_nova,
         mape_anterior, mape_novo, melhoria_pct, duracao_seg, detalhes)
        VALUES(?,?,?,?,?,?,?,?,?)
    """, (trigger, status, version_ant, version_novo,
          mape_ant, mape_novo, melhoria, round(duracao,2), detalhes))
    conn.commit()
    conn.close()


def executar_retreinamento(trigger="scheduled", nivel="categoria", forcar=False):
    """
    Executa pipeline completo de retreinamento.
    Compara com modelo atual e só promove se melhorar.
    """
    from src.features.feature_engineering import criar_features_categoria, criar_features_sku
    from src.models.trainer import treinar

    _log(f"Iniciando retreinamento [{nivel}] — trigger: {trigger}")
    t0 = time.time()

    _, meta_atual = get_production_model(nivel)
    mape_atual    = meta_atual["mape"] if meta_atual else None
    version_atual = meta_atual["version"] if meta_atual else None

    try:
        # 1. Recalcular features
        _log("Recalculando features...")
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../data/processed"), exist_ok=True)
        if nivel == "categoria":
            df = criar_features_categoria()
            df.to_csv(os.path.join(os.path.dirname(__file__), "../../data/processed/features_categoria.csv"), index=False)
        else:
            df = criar_features_sku()
            df.to_csv(os.path.join(os.path.dirname(__file__), "../../data/processed/features_sku.csv"), index=False)

        # 2. Treinar novo modelo
        _log("Treinando novo modelo...")
        version_novo, metricas = treinar(nivel, forcar_promocao=forcar)

        duracao = time.time() - t0
        _log(f"Retreinamento concluído em {duracao:.1f}s — MAPE: {metricas['mape']:.1f}%")

        _registrar_log_retreino(
            trigger, "success", version_atual, version_novo,
            mape_atual, metricas["mape"], duracao
        )

        # 3. Rollback automático se piorou significativamente
        if mape_atual and metricas["mape"] > mape_atual * 1.10:
            _log(f"⚠️  Novo MAPE ({metricas['mape']:.1f}%) > 110% do atual ({mape_atual:.1f}%) — rollback", "WARN")
            rollback(nivel)

        return True, version_novo

    except Exception as e:
        duracao = time.time() - t0
        _log(f"❌ Erro no retreinamento: {e}", "ERROR")
        _registrar_log_retreino(trigger, "failed", version_atual, None,
                                mape_atual, None, duracao, str(e))
        return False, None


def verificar_e_retreinar():
    """Verifica drift e retreina se necessário (job de emergência)."""
    for nivel in ["categoria"]:
        retratar, motivo = precisa_retreinar(nivel)
        if retratar:
            _log(f"🚨 Drift detectado [{nivel}]: {motivo}", "WARN")
            executar_retreinamento(trigger="drift_alert", nivel=nivel)
        else:
            _log(f"✅ Modelo [{nivel}] estável: {motivo}")


def job_semanal():
    """Simula job semanal de retreinamento (como Airflow DAG)."""
    _log("="*50)
    _log("JOB SEMANAL — Retreinamento Programado")
    _log("="*50)
    executar_retreinamento(trigger="scheduled", nivel="categoria", forcar=False)


def iniciar_scheduler(intervalo_horas=1, apenas_uma_vez=False):
    """
    Inicia o scheduler em thread separada.
    Em produção seria substituído por Airflow/cron.
    intervalo_horas: verificação de drift
    """
    _log("🕒 Scheduler iniciado")
    _log(f"   Verificação de drift a cada {intervalo_horas}h")
    _log("   Retreinamento semanal: toda segunda-feira")

    def loop():
        ciclo = 0
        while True:
            ciclo += 1
            _log(f"--- Ciclo #{ciclo} ---")

            # Verificar drift
            verificar_e_retreinar()

            # Retreinamento semanal (simula segunda-feira)
            if ciclo % (7 * 24 // intervalo_horas) == 0:
                job_semanal()

            if apenas_uma_vez:
                break

            time.sleep(intervalo_horas * 3600)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def historico_retreinamentos():
    return get_conn().execute("""
        SELECT trigger, status, version_anterior, version_nova,
               mape_anterior, mape_novo, melhoria_pct, duracao_seg, executado_em
        FROM retraining_log
        ORDER BY executado_em DESC
        LIMIT 20
    """).fetchall()


if __name__ == "__main__":
    _log("Executando verificação de drift única...")
    verificar_e_retreinar()

    hist = historico_retreinamentos()
    print(f"\nÚltimos retreinamentos: {len(hist)}")
    for h in hist:
        print(f"  {h['executado_em']} | {h['trigger']:12} | {h['status']:7} | MAPE {h['mape_anterior']} → {h['mape_novo']}")
