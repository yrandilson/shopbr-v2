"""
ShopBR v2 — Gerador de Dados Avançado
Simula dados de nível SKU com: promoções, estoque, índice de concorrência, campanhas.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import init_db, get_conn, executemany

np.random.seed(42)

# ── Catálogo de SKUs ──────────────────────────────────────────────────────────
SKUS = {
    # (categoria, preco_base, volume_base, trend_diario)
    "ELE-001": ("Eletrônicos", "Smartphone Entry",    899.0,  85, 0.00030),
    "ELE-002": ("Eletrônicos", "Notebook 15\"",      2499.0,  32, 0.00020),
    "ELE-003": ("Eletrônicos", "Fone Bluetooth",      189.0, 140, 0.00040),
    "ELE-004": ("Eletrônicos", "Smartwatch",          599.0,  60, 0.00035),
    "MOD-001": ("Moda",        "Tênis Casual",        189.0, 145, 0.00015),
    "MOD-002": ("Moda",        "Camiseta Básica",      49.9, 310, 0.00010),
    "MOD-003": ("Moda",        "Calça Jeans",         149.0, 120, 0.00012),
    "MOD-004": ("Moda",        "Vestido Floral",      129.0,  95, 0.00018),
    "CAS-001": ("Casa & Jardim","Jogo de Cama",       149.0,  72, 0.00020),
    "CAS-002": ("Casa & Jardim","Panela Antiaderente",  89.9, 110, 0.00015),
    "CAS-003": ("Casa & Jardim","Luminária LED",        69.9,  90, 0.00025),
    "BEL-001": ("Beleza",      "Hidratante Corporal",  45.9, 210, 0.00045),
    "BEL-002": ("Beleza",      "Protetor Solar FPS50", 39.9, 185, 0.00040),
    "BEL-003": ("Beleza",      "Perfume 100ml",       189.0,  80, 0.00030),
    "ESP-001": ("Esportes",    "Whey Protein 1kg",    129.0,  95, 0.00025),
    "ESP-002": ("Esportes",    "Tênis Running",       349.0,  55, 0.00020),
    "ESP-003": ("Esportes",    "Corda de Pular",       34.9, 130, 0.00015),
}

SAZON_MES = {1:0.72,2:0.68,3:0.85,4:0.90,5:1.05,6:1.02,
             7:0.95,8:1.00,9:0.92,10:0.97,11:1.55,12:1.70}
SAZON_DOW = {0:0.88,1:0.85,2:0.90,3:0.93,4:1.05,5:1.20,6:1.18}

def feriado_mult(data):
    m, d = data.month, data.day
    if m == 11 and 20 <= d <= 30: return 2.8
    if m == 12 and 15 <= d <= 31: return 1.8
    if m == 5  and  8 <= d <= 14: return 1.5
    if m == 6  and  8 <= d <= 14: return 1.3
    if m == 8  and  8 <= d <= 14: return 1.3
    if m == 2  and  d <= 5:       return 0.65   # Carnaval
    return 1.0

def gerar_campanha():
    """Gera períodos de campanha de marketing por SKU."""
    campanhas = {}
    datas = pd.date_range("2022-01-01", "2024-03-31")
    for sku in SKUS:
        ativas = set()
        # ~6 campanhas por ano
        for _ in range(14):
            start = np.random.choice(datas)
            dur   = np.random.randint(3, 15)
            for i in range(dur):
                d = start + pd.Timedelta(days=int(i))
                if d in pd.DatetimeIndex(datas):
                    ativas.add(d)
        campanhas[sku] = ativas
    return campanhas

def gerar_concorrencia():
    """Índice de preço da concorrência (0.8 = concorrente 20% mais barato)."""
    datas = pd.date_range("2022-01-01", "2024-03-31")
    idx   = {}
    for sku in SKUS:
        base  = 1.0
        serie = [base]
        for _ in range(len(datas)-1):
            base += np.random.normal(0, 0.005)
            base  = np.clip(base, 0.7, 1.3)
            serie.append(base)
        idx[sku] = dict(zip(datas, serie))
    return idx

def gerar():
    print("🏗️  Inicializando banco de dados...")
    init_db()

    campanhas   = gerar_campanha()
    concorrencia = gerar_concorrencia()
    datas = pd.date_range("2022-01-01", "2024-03-31")

    # ── Inserir produtos ───────────────────────────────────────────────────────
    conn = get_conn()
    conn.execute("DELETE FROM products")
    conn.execute("DELETE FROM sales")
    conn.commit()

    produtos_rows = [
        (sku, info[1], info[0], info[2])
        for sku, info in SKUS.items()
    ]
    conn.executemany("INSERT OR REPLACE INTO products(sku,nome,categoria,preco_base) VALUES(?,?,?,?)", produtos_rows)
    conn.commit()

    # ── Gerar vendas ───────────────────────────────────────────────────────────
    print("📊 Gerando vendas por SKU...")
    registros = []

    for sku, (cat, nome, preco_base, vol_base, trend) in SKUS.items():
        estoque = int(vol_base * 30)  # estoque inicial = 30 dias de venda

        for i, data in enumerate(datas):
            mult_saz   = SAZON_MES[data.month]
            mult_dow   = SAZON_DOW[data.dayofweek]
            mult_fer   = feriado_mult(data)
            mult_trend = 1 + trend * i
            mult_camp  = 1.35 if data in campanhas[sku] else 1.0
            idx_conc   = concorrencia[sku][data]
            # Concorrente barato reduz nossas vendas
            mult_conc  = 1 + (1 - idx_conc) * (-0.5)

            ruido    = np.random.lognormal(0, 0.10)
            unidades = int(vol_base * mult_saz * mult_dow * mult_fer * mult_trend * mult_camp * mult_conc * ruido)
            unidades = max(1, min(unidades, estoque))  # limitado pelo estoque

            promo = 1 if data in campanhas[sku] else 0
            preco = preco_base * np.random.uniform(0.88 if promo else 0.95, 0.98 if promo else 1.05)

            # Reposição de estoque: quando cai abaixo de 5 dias, repõe
            estoque -= unidades
            if estoque < vol_base * 5:
                estoque += int(vol_base * 20 * np.random.uniform(0.8, 1.2))

            registros.append((
                str(data.date()), sku, cat, unidades,
                round(preco, 2), round(unidades * preco, 2),
                promo, estoque, round(idx_conc, 4)
            ))

    conn.executemany("""
        INSERT INTO sales(data,sku,categoria,unidades,preco_venda,receita,
                          promocao,estoque_ini,indice_concorr)
        VALUES(?,?,?,?,?,?,?,?,?)
    """, registros)
    conn.commit()
    conn.close()

    total = len(registros)
    print(f"✅ {total:,} registros inseridos no banco")
    print(f"   SKUs: {len(SKUS)} | Datas: {len(datas)} dias | Categorias: 5")
    return total


if __name__ == "__main__":
    gerar()
