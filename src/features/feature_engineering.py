"""
ShopBR v2 — Feature Engineering Avançada
Suporte a nível CATEGORIA e SKU com features extras: promoção, estoque, concorrência.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import query_df


def _base_calendar(df):
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df["dia_semana"]    = df["data"].dt.dayofweek
    df["mes"]           = df["data"].dt.month
    df["semana_ano"]    = df["data"].dt.isocalendar().week.astype(int)
    df["dia_mes"]       = df["data"].dt.day
    df["trimestre"]     = df["data"].dt.quarter
    df["is_fim_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["mes_sin"]       = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"]       = np.cos(2 * np.pi * df["mes"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["dia_semana"] / 7)

    def flag_fer(row):
        m, d = row["mes"], row["dia_mes"]
        if m == 11 and 20 <= d <= 30: return 1
        if m == 12 and 15 <= d <= 31: return 1
        if m == 5  and  8 <= d <= 14: return 1
        if m == 6  and  8 <= d <= 14: return 1
        if m == 8  and  8 <= d <= 14: return 1
        if m == 2  and  d <= 5:       return 1
        return 0

    df["is_feriado"] = df.apply(flag_fer, axis=1)
    return df


def _ts_features(grupo, target_col, id_col):
    """Lags, rolling, diff para uma série temporal de um grupo."""
    serie = grupo[target_col]

    for lag in [1, 7, 14, 21, 28, 35]:
        grupo[f"lag_{lag}d"] = serie.shift(lag)

    for w in [7, 14, 28]:
        s = serie.shift(1)
        grupo[f"roll_mean_{w}d"] = s.rolling(w).mean()
        grupo[f"roll_std_{w}d"]  = s.rolling(w).std()
        grupo[f"roll_max_{w}d"]  = s.rolling(w).max()
        grupo[f"roll_min_{w}d"]  = s.rolling(w).min()
        grupo[f"roll_med_{w}d"]  = s.rolling(w).median()

    grupo["diff_1d"]  = serie.shift(1).diff(1)
    grupo["diff_7d"]  = serie.shift(1).diff(7)
    grupo["diff_28d"] = serie.shift(1).diff(28)
    grupo["trend_idx"] = np.arange(len(grupo))

    # Velocidade de crescimento recente vs histórico
    grupo["aceleracao_7_28"] = (
        grupo["roll_mean_7d"] / (grupo["roll_mean_28d"] + 1e-9)
    )

    # Rolling da receita
    grupo["roll_receita_7d"]  = grupo["receita"].shift(1).rolling(7).mean()
    grupo["roll_receita_28d"] = grupo["receita"].shift(1).rolling(28).mean()

    return grupo


def criar_features_categoria():
    """Features agregadas por CATEGORIA/DIA."""
    df = query_df("""
        SELECT data, categoria,
               SUM(unidades) as unidades,
               SUM(receita)  as receita,
               AVG(preco_venda) as preco_medio,
               AVG(indice_concorr) as indice_concorr,
               MAX(promocao) as tem_promocao,
               AVG(estoque_ini) as estoque_medio
        FROM sales
        GROUP BY data, categoria
        ORDER BY categoria, data
    """)

    df = _base_calendar(df)

    resultado = []
    for cat, grp in df.groupby("categoria"):
        grp = grp.copy().reset_index(drop=True)
        grp = _ts_features(grp, "unidades", "categoria")

        # Features extras de negócio
        grp["lag_concorr_7d"] = grp["indice_concorr"].shift(1).rolling(7).mean()
        grp["promo_lag_1d"]   = grp["tem_promocao"].shift(1)
        grp["promo_lag_7d"]   = grp["tem_promocao"].shift(1).rolling(7).sum()

        resultado.append(grp)

    out = pd.concat(resultado).reset_index(drop=True)
    out = pd.get_dummies(out, columns=["categoria"], prefix="cat")
    return out


def criar_features_sku():
    """Features por SKU/DIA."""
    df = query_df("""
        SELECT s.data, s.sku, s.categoria, s.unidades, s.receita,
               s.preco_venda, s.indice_concorr, s.promocao, s.estoque_ini,
               p.preco_base
        FROM sales s
        JOIN products p ON s.sku = p.sku
        ORDER BY s.sku, s.data
    """)

    df = _base_calendar(df)

    # Relativo ao preço base
    df["desconto_pct"] = (df["preco_base"] - df["preco_venda"]) / df["preco_base"]

    resultado = []
    for sku, grp in df.groupby("sku"):
        grp = grp.copy().reset_index(drop=True)
        grp = _ts_features(grp, "unidades", "sku")
        grp["lag_concorr_7d"] = grp["indice_concorr"].shift(1).rolling(7).mean()
        grp["promo_lag_1d"]   = grp["promocao"].shift(1)
        grp["estoque_lag_1d"] = grp["estoque_ini"].shift(1)
        resultado.append(grp)

    out = pd.concat(resultado).reset_index(drop=True)
    out = pd.get_dummies(out, columns=["categoria"], prefix="cat")
    return out


if __name__ == "__main__":
    print("Feature engineering — categoria...")
    df_cat = criar_features_categoria()
    print(f"  Shape: {df_cat.shape}")

    print("Feature engineering — SKU...")
    df_sku = criar_features_sku()
    print(f"  Shape: {df_sku.shape}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "../../data/processed"), exist_ok=True)
    base = os.path.join(os.path.dirname(__file__), "../../data/processed")
    df_cat.to_csv(f"{base}/features_categoria.csv", index=False)
    df_sku.to_csv(f"{base}/features_sku.csv", index=False)
    print("✅ Features salvas")
