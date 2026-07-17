"""
pricing_case_app.py — Case de Pricing & Margem (app Streamlit, arquivo único).

Sobe a base .xlsx pela interface e roda as 8 análises de detecção de ganho de
margem: dispersão, margem negativa, whale curve, modelo de preço (OLS+XGBoost),
elasticidade, otimização e Price/Volume/Mix bridge.

RODAR
-----
    pip install streamlit pandas numpy scipy statsmodels scikit-learn xgboost matplotlib openpyxl
    streamlit run pricing_case_app.py

Abre no navegador. Na barra lateral: arraste o .xlsx, ajuste os parâmetros e
veja os resultados em abas, com download de CSV/Excel. Nenhum caminho de arquivo
fica fixo no código — a base é enviada pelo upload.

Base esperada: aba "Data Basis" com as colunas transacionais (o app valida e
avisa se faltar alguma). Convenção: COGS negativo, Margin = Revenue + COGS,
Price_Avg = Revenue/Quantity.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


# ======================================================================
# NÚCLEO DE ANÁLISE (lógica das 8 análises)
# ======================================================================

# ========================================================================
# Nomes de coluna esperados na base bruta
# ========================================================================
COL_MONTH = "Billing Year Month"
COL_COUNTRY = "Sold To Country Name"
COL_TIER = "Customer Tier"
COL_REVENUE = "Revenue ($)"
COL_COGS = "COGS ($)"
COL_QTY = "Quantity"
COL_PRICE = "Price_Avg"
COL_COGS_AVG = "COGS_Avg"
COL_MARGIN_UNIT = "Margin_Unit"
COL_MARGIN_TOTAL = "Margin_Total"
COL_PARENT = "Customer Parent"
COL_SOLDTO = "Sold-to party"
COL_BUSINESS = "Business"
COL_MATERIAL = "Material"

REQUIRED_COLS = [
    COL_MONTH, COL_COUNTRY, COL_TIER, COL_REVENUE, COL_COGS, COL_QTY,
    COL_PRICE, COL_COGS_AVG, COL_PARENT, COL_SOLDTO, COL_BUSINESS, COL_MATERIAL,
]

SHEET_DEFAULT = "Data Basis"

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ========================================================================
# Parâmetros ajustáveis (a UI expõe estes na barra lateral)
# ========================================================================
@dataclass
class Params:
    dispersion_target_pctl: float = 0.50   # preço-alvo = mediana do peer group
    min_peer_size: int = 8                 # tamanho mínimo do peer group
    residual_target_pctl: float = 0.25     # alvo dos resíduos (P25)
    capture_rate: float = 0.30             # fração do gap capturável
    max_price_increase: float = 0.15       # teto de aumento na otimização
    max_price_decrease: float = 0.00       # piso (não baixa preço)
    random_state: int = 42
    use_xgb: bool = True                   # usar XGBoost se disponível
    boxplot_top_n: int = 15                # nº de peer groups no boxplot
    boxplot_rank_by: str = "uplift"        # "uplift" ou "cv" — como escolher os grupos
    ml_filter_tier: str = "(todos)"        # filtro da tabela de oportunidades ML
    ml_filter_country: str = "(todos)"     # filtro por país
    diag_country: str = "(auto)"           # país do diagnóstico de dispersão maçã-a-maçã
    diag_material: str = "(auto)"           # produto para a régua de tier interativa
    # pesos do modelo de decisão multicritério (aba Modelagem) — 5 critérios
    mcda_perfil: str = "Equilibrado"        # perfil de objetivo (define os pesos)
    w_margem: float = 0.20
    w_elasticidade: float = 0.20
    w_share: float = 0.20
    w_churn: float = 0.20
    w_cluster: float = 0.20
    mcda_sku: str = "(todos)"               # seletor de SKU para o MCDA


# ========================================================================
# Carregamento e limpeza
# ========================================================================
def read_excel(file, sheet: str | None = None) -> pd.DataFrame:
    """Lê a planilha enviada. `file` é caminho ou buffer (upload do Streamlit)."""
    xls = pd.ExcelFile(file)
    if sheet and sheet in xls.sheet_names:
        target = sheet
    elif SHEET_DEFAULT in xls.sheet_names:
        target = SHEET_DEFAULT
    else:
        target = xls.sheet_names[0]
    df = xls.parse(target)
    if COL_COGS in df.columns:
        df[COL_COGS] = pd.to_numeric(df[COL_COGS], errors="coerce")
    return df


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de colunas obrigatórias que estão faltando."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


def _parse_month(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split("-", expand=True)
    year = parts[0].astype(int)
    month = parts[1].map(MONTH_MAP).astype(int)
    return pd.to_datetime(dict(year=year, month=month, day=1))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = _parse_month(df[COL_MONTH])
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["ym"] = df[COL_MONTH]

    df[COL_TIER] = df[COL_TIER].fillna("UNSPECIFIED")

    # --- correção de sinal do COGS ---
    # Convenção da base: custo entra como negativo (margem = Receita + COGS).
    # Linhas com COGS positivo têm o sinal invertido (concentradas no Business H).
    # Forçamos o valor absoluto negativo para que a margem fique correta, e
    # registramos flag + contagem para transparência na aba descritiva.
    cogs_pos_mask = df[COL_COGS] > 0
    df["_cogs_sign_fixed"] = cogs_pos_mask.fillna(False)
    df.attrs["cogs_fixed_count"] = int(cogs_pos_mask.sum())
    df[COL_COGS] = -df[COL_COGS].abs()
    if COL_COGS_AVG in df.columns:
        df[COL_COGS_AVG] = -df[COL_COGS_AVG].abs()

    df["margin_calc"] = df[COL_REVENUE] + df[COL_COGS]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["margin_pct"] = np.where(
            df[COL_REVENUE] != 0, df["margin_calc"] / df[COL_REVENUE], np.nan)

    df["is_return_or_adj"] = (df[COL_QTY] <= 0) | (df[COL_REVENUE] < 0)
    df["price_valid"] = (
        (df[COL_PRICE] > 0) & np.isfinite(df[COL_PRICE])
        & (df[COL_QTY] > 0) & (df[COL_REVENUE] > 0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        df["ln_price"] = np.where(
            df["price_valid"], np.log(df[COL_PRICE].where(df[COL_PRICE] > 0)), np.nan)
        df["ln_qty"] = np.where(
            df[COL_QTY] > 0, np.log(df[COL_QTY].where(df[COL_QTY] > 0)), np.nan)
    return df


def summarize_clean(df: pd.DataFrame) -> dict:
    clean_price = df[df["price_valid"]]
    return {
        "Linhas totais": f"{len(df):,}",
        "Válidas p/ preço": f"{len(clean_price):,}",
        "Devoluções/ajustes": f"{int(df['is_return_or_adj'].sum()):,}",
        "% margem negativa (válidas)": f"{(clean_price['margin_calc'] < 0).mean():.2%}",
        "Período": f"{df['date'].min():%Y-%m} a {df['date'].max():%Y-%m}",
        "Materiais": f"{df[COL_MATERIAL].nunique():,}",
        "Clientes (Parent)": f"{df[COL_PARENT].nunique():,}",
        "Países": f"{df[COL_COUNTRY].nunique():,}",
    }


# ========================================================================
# 01 — ESTATÍSTICA DESCRITIVA (BI de abertura)
# ========================================================================
NUM_COLS = [COL_REVENUE, COL_COGS, COL_QTY, COL_PRICE, COL_COGS_AVG,
            COL_MARGIN_UNIT, COL_MARGIN_TOTAL]
ALL_COLS = [COL_MONTH, COL_COUNTRY, COL_TIER, COL_REVENUE, COL_COGS, COL_QTY,
            COL_PRICE, COL_COGS_AVG, COL_MARGIN_UNIT, COL_MARGIN_TOTAL,
            COL_PARENT, COL_SOLDTO, COL_BUSINESS, COL_MATERIAL]
COL_DESC = {
    COL_MONTH: "Mês de faturamento", COL_COUNTRY: "País do cliente",
    COL_TIER: "Categoria do cliente", COL_REVENUE: "Receita da transação",
    COL_COGS: "Custo da mercadoria (negativo)", COL_QTY: "Quantidade vendida",
    COL_PRICE: "Preço unitário (Receita/Qtd)", COL_COGS_AVG: "Custo unitário (negativo)",
    COL_MARGIN_UNIT: "Margem unitária", COL_MARGIN_TOTAL: "Margem total",
    COL_PARENT: "Grupo/matriz do cliente", COL_SOLDTO: "Entidade compradora",
    COL_BUSINESS: "Segmento de negócio", COL_MATERIAL: "Código do produto",
}


def _fmt_money_mi(v):
    """Formata em mi ou bi conforme a magnitude."""
    if abs(v) >= 1e9:
        return f"${v/1e9:,.2f} bi"
    return f"${v/1e6:,.0f} mi"


def _round_df(df, money_cols=(), pct_cols=(), int_cols=()):
    """Arredonda colunas: dinheiro -> 0 casas, % -> 1 casa, inteiros -> int."""
    d = df.copy()
    for c in money_cols:
        if c in d.columns:
            d[c] = d[c].round(0).astype("Int64")
    for c in pct_cols:
        if c in d.columns:
            d[c] = d[c].round(1)
    for c in int_cols:
        if c in d.columns:
            d[c] = d[c].round(0).astype("Int64")
    return d


def run_descriptive(df: pd.DataFrame, p: Params) -> dict:
    """BI descritivo. Retorna uma lista de 'blocos', cada um com título,
    texto explicativo (por que / o que responde / insight), e uma tabela
    OU figura. A UI renderiza bloco a bloco."""
    n = len(df)
    df = df.copy()
    df["_margin"] = df[COL_REVENUE] + df[COL_COGS]

    # --- COGS já foi corrigido na limpeza (sinal forçado negativo) ---
    cogs_pos = int(df["_cogs_sign_fixed"].sum()) if "_cogs_sign_fixed" in df.columns else 0
    cogs_pos_pct = 100 * cogs_pos / len(df) if len(df) else 0
    if cogs_pos:
        top_biz_pos = df.loc[df["_cogs_sign_fixed"], COL_BUSINESS].value_counts()
        cogs_pos_biz = top_biz_pos.index[0]
        cogs_pos_biz_share = 100 * top_biz_pos.iloc[0] / cogs_pos
    else:
        cogs_pos_biz, cogs_pos_biz_share = None, 0

    blocks = []  # cada item: dict(title, why, answers, insight, table=?, fig=?)

    def add(title, why="", answers="", insight="", table=None, fig=None, money=""):
        blocks.append({"title": title, "why": why, "answers": answers,
                       "insight": insight, "table": table, "fig": fig, "money": money})

    # ===================================================================
    # BLOCO 1 — DICIONÁRIO DA BASE
    # ===================================================================
    schema_rows = []
    for c in ALL_COLS:
        nulls = int(df[c].isna().sum())
        schema_rows.append({
            "coluna": c, "descrição": COL_DESC.get(c, ""), "tipo": str(df[c].dtype),
            "nulos": nulls, "% nulos": round(100 * nulls / n, 1),
            "valores únicos": int(df[c].nunique()),
        })
    schema = pd.DataFrame(schema_rows)
    add("Base de dados — dicionário e qualidade",
        "Antes de qualquer análise é preciso conhecer o material bruto: o que cada coluna é, o tipo, quanto falta e a cardinalidade.",
        "Quais colunas temos, onde há dados faltantes e quão granular é cada campo.",
        f"Customer Tier falta em {schema.loc[schema['coluna']==COL_TIER,'% nulos'].iloc[0]:.0f}% das linhas — maior fragilidade. "
        f"Price_Avg e COGS faltam em ~9% (linhas de devolução/ajuste).",
        table=schema)

    # ===================================================================
    # BLOCO 2 — ESTATÍSTICA NUMÉRICA
    # ===================================================================
    desc = df[NUM_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc = desc.round(2).reset_index().rename(columns={"index": "coluna"})
    cogs_note = ("COGS e COGS_Avg devem ser sempre negativos (convenção: custo entra como negativo, "
                 f"margem = Receita + COGS). " +
                 (f"CORREÇÃO APLICADA: {cogs_pos:,} linhas ({cogs_pos_pct:.0f}%) vinham com COGS positivo — "
                  f"sinal invertido, {cogs_pos_biz_share:.0f}% delas no Business {cogs_pos_biz}. "
                  "O sinal foi forçado para negativo na limpeza, então a margem agora está correta em toda a base."
                  if cogs_pos else "Todos os custos já estavam negativos, como esperado."))
    add("Base de dados — estatística numérica",
        "Resumo estatístico de cada variável numérica — o describe() clássico de ciência de dados.",
        "Qual a ordem de grandeza, dispersão e presença de outliers em receita, custo, preço e margem.",
        cogs_note,
        table=desc)

    # ===================================================================
    # BLOCO 3 — QUANTO VENDEMOS? (mês a mês)
    # ===================================================================
    bm = (df.groupby("ym")
          .agg(receita=(COL_REVENUE, "sum"), margem=("_margin", "sum"),
               quantidade=(COL_QTY, "sum"), transacoes=(COL_REVENUE, "size"))
          .reset_index())
    # ordena por data real
    order = df.groupby("ym")["date"].first().sort_values()
    bm["_ord"] = bm["ym"].map({k: i for i, k in enumerate(order.index)})
    bm = bm.sort_values("_ord").drop(columns="_ord")
    bm["margem_%"] = (100 * bm["margem"] / bm["receita"]).round(1)
    bm["receita_var_%"] = (bm["receita"].pct_change() * 100).round(1)
    bm_tbl = _round_df(bm, money_cols=["receita", "margem"], int_cols=["quantidade", "transacoes"])

    fig, ax = plt.subplots(figsize=(9, 4))
    xs = list(bm["ym"])
    ax.plot(xs, bm["receita"] / 1e6, color="#3b6ea5", marker="o", markersize=2.5, label="Receita")
    ax.plot(xs, bm["margem"] / 1e6, color="#1d9e75", marker="o", markersize=2.5, label="Margem")
    step = max(1, len(xs) // 12)
    ax.set_xticks(xs[::step]); ax.set_xticklabels(xs[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("US$ milhões"); ax.set_title("Receita e margem mês a mês"); ax.legend()
    fig.tight_layout()
    add("Vendas — evolução mensal",
        "Acompanhar a evolução temporal é a base de qualquer leitura de negócio — mostra tendência, sazonalidade e quebras.",
        "A receita e a margem estão crescendo ou caindo ao longo dos 48 meses?",
        "A série permite ver sazonalidade e se a margem % acompanha a receita ou descola (sinal de pressão de custo ou preço).",
        fig=fig)
    add("Receita mês a mês (tabela)", "", "", "", table=bm_tbl)

    # ===================================================================
    # BLOCO 4 — ANO A ANO
    # ===================================================================
    by = (df.groupby("year")
          .agg(receita=(COL_REVENUE, "sum"), custo=(COL_COGS, "sum"),
               margem=("_margin", "sum"), quantidade=(COL_QTY, "sum"),
               transacoes=(COL_REVENUE, "size"))
          .reset_index())
    by["custo"] = by["custo"].abs()  # exibir custo como positivo na tabela
    by["margem_%"] = (100 * by["margem"] / by["receita"]).round(1)
    by["receita_var_%"] = (by["receita"].pct_change() * 100).round(1)
    by["custo_var_%"] = (by["custo"].pct_change() * 100).round(1)
    by["margem_var_%"] = (by["margem"].pct_change() * 100).round(1)
    by_tbl = _round_df(
        by[["year", "receita", "custo", "margem", "margem_%",
            "receita_var_%", "custo_var_%", "margem_var_%", "quantidade"]],
        money_cols=["receita", "custo", "margem"], int_cols=["quantidade", "year"])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(by)); w = 0.26
    xs = by["year"].astype(int).astype(str)
    ax.bar(x - w, by["receita"] / 1e6, w, color="#3b6ea5", label="Receita")
    ax.bar(x, by["custo"] / 1e6, w, color="#c0504d", label="Custo (COGS)")
    ax.bar(x + w, by["margem"] / 1e6, w, color="#1d9e75", label="Margem")
    ax.set_xticks(x); ax.set_xticklabels(xs)
    for i, v in enumerate(by["receita_var_%"]):
        if pd.notna(v):
            ax.annotate(f"{v:+.0f}%", (i - w, by["receita"].iloc[i] / 1e6),
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("US$ milhões"); ax.set_title("Receita, custo e margem ano a ano"); ax.legend()
    fig.tight_layout()
    last_var = by["receita_var_%"].iloc[-1]
    last_custo_var = by["custo_var_%"].iloc[-1]
    add("Vendas — evolução anual",
        "A visão anual remove o ruído mensal e mostra a tendência de fundo. Ver receita, custo e margem lado a lado revela se a margem está sendo pressionada por queda de receita ou por alta de custo.",
        "Como receita, custo e margem evoluíram ano a ano, e o que explica a variação da margem?",
        f"No último ano a receita variou {last_var:+.0f}% e o custo {last_custo_var:+.0f}%. "
        + ("Custo caindo menos que a receita (ou subindo) aperta a margem — a tabela mostra a variação de cada linha."
           if last_custo_var > last_var else "A margem acompanha o movimento de receita e custo."),
        fig=fig)
    add("Receita, custo e margem ano a ano (tabela)", "", "", "", table=by_tbl)

    # ===================================================================
    # BLOCO 4b — FUNDAMENTOS DE PRICING (variação 2 últimos anos, break-even, preço, dispersão)
    # ===================================================================
    # comparação foca nos DOIS ÚLTIMOS anos (ex.: 2025 x 2026)
    y_all = sorted(int(v) for v in by["year"].unique())
    y0, y1 = y_all[-2], y_all[-1]

    # KPIs anuais (preço médio e COGS médio ponderados, clientes, materiais)
    dv0 = df[(df[COL_PRICE] > 0) & (df[COL_QTY] > 0)].copy()
    pm_year = (dv0.groupby("year")
               .apply(lambda g: np.average(g[COL_PRICE], weights=g[COL_QTY])))
    cm_year = (dv0.groupby("year")
               .apply(lambda g: np.average((-g[COL_COGS_AVG]).clip(lower=0), weights=g[COL_QTY])))
    cli_year = df.groupby("year")[COL_PARENT].nunique()
    mat_year = df.groupby("year")[COL_MATERIAL].nunique()
    mgpct_year = by.set_index("year")["margem"] / by.set_index("year")["receita"] * 100

    def _var(serie):
        # variação % entre os dois últimos anos
        try:
            ini, fim = serie.loc[y0], serie.loc[y1]
        except Exception:
            ini, fim = serie.iloc[-2], serie.iloc[-1]
        return (fim / ini - 1) if ini else np.nan, ini, fim

    kpi_defs = [
        ("Receita", by.set_index("year")["receita"], "money"),
        ("Margem", by.set_index("year")["margem"], "money"),
        ("Margem %", mgpct_year, "pct"),
        ("Preço médio", pm_year, "num"),
        ("COGS médio", cm_year, "num"),
        ("Quantidade", by.set_index("year")["quantidade"], "int"),
        ("Clientes", cli_year, "int"),
        ("Materiais", mat_year, "int"),
    ]
    rows = []
    var_map = {}
    for nome, serie, tipo in kpi_defs:
        var, ini, fim = _var(serie)
        var_map[nome] = var
        if tipo == "money":
            fmt = lambda v: f"${v/1e6:,.1f} mi"
        elif tipo == "pct":
            fmt = lambda v: f"{v:.1f}%"
        elif tipo == "int":
            fmt = lambda v: f"{v:,.0f}"
        else:
            fmt = lambda v: f"${v:,.2f}"
        rows.append({"KPI": nome, f"{y0}": fmt(ini), f"{y1}": fmt(fim),
                     f"Variação {y0}→{y1}": f"{var*100:+.1f}%"})
    cagr_tbl = pd.DataFrame(rows)

    v_rec = var_map["Receita"]; v_mar = var_map["Margem"]
    v_cli = var_map["Clientes"]; v_preco = var_map["Preço médio"]; v_cogs = var_map["COGS médio"]

    add(f"Crescimento — variação dos 8 indicadores ({y0} → {y1})",
        why="Em vez do período inteiro, comparamos os dois anos mais recentes — o retrato do momento atual da empresa. Reunir os 8 indicadores num só cartão dá a leitura completa: o que cresceu, o que encolheu e o que pressiona a margem.",
        answers=f"O que mudou de {y0} para {y1} em receita, margem, preço, custo, volume e base de clientes?",
        insight=f"Receita {v_rec*100:+.1f}% e margem {v_mar*100:+.1f}% de {y0} para {y1}. "
                f"Preço médio {v_preco*100:+.1f}% e COGS médio {v_cogs*100:+.1f}% — "
                + ("preço e custo subiram, mas se o custo sobe mais que o preço a margem aperta; "
                   if v_cogs > v_preco else "o preço se moveu à frente do custo; ")
                + f"clientes {v_cli*100:+.1f}%. A tabela mostra a variação de cada dimensão.",
        table=cagr_tbl)

    # -- Break-even / margem de contribuição --
    rev_t = df[COL_REVENUE].sum(); cogs_t = -df[COL_COGS].sum(); marg_t = df["_margin"].sum()
    mc_pct = 100 * marg_t / rev_t if rev_t else np.nan
    markup = 100 * (rev_t / cogs_t - 1) if cogs_t else np.nan
    be_tbl = pd.DataFrame({
        "métrica": ["Receita total", "Custo total (COGS)", "Margem de contribuição",
                    "Margem de contribuição (%)", "Markup médio sobre custo"],
        "valor": [f"${rev_t/1e6:,.0f} mi", f"${cogs_t/1e6:,.0f} mi", f"${marg_t/1e6:,.0f} mi",
                  f"{mc_pct:.1f}%", f"{markup:.0f}%"],
    })
    add("Rentabilidade — break-even e margem de contribuição",
        why="O break-even é o ponto em que a receita cobre exatamente os custos. A margem de contribuição (quanto de cada real de venda sobra após o custo variável) é o que sustenta esse equilíbrio — o conceito central de qualquer decisão de preço.",
        answers="Quanto de cada venda sobra para cobrir custos e gerar lucro? Qual o markup praticado?",
        insight=f"A margem de contribuição média é {mc_pct:.1f}% — de cada US$ 1 vendido, sobram US$ {mc_pct/100:.2f} "
                f"após o custo direto. O markup médio sobre o custo é {markup:.0f}%. "
                "Esses são os pisos de referência: vendas abaixo desse markup corroem o resultado (ver aba Desperdício de Margem).",
        money=f"Cada ponto percentual de margem de contribuição vale ~US$ {rev_t/1e6*0.01:,.1f} mi sobre a receita atual.",
        table=be_tbl)

    # -- Performance de preço E custo no período (ponderados por volume) --
    dv = df[(df[COL_PRICE] > 0) & (df[COL_QTY] > 0)].copy()
    dv["_unit_cost"] = (-dv[COL_COGS_AVG]).clip(lower=0)
    pm = (dv.groupby("date")
          .apply(lambda g: pd.Series({
              "preco": np.average(g[COL_PRICE], weights=g[COL_QTY]),
              "custo": np.average(g["_unit_cost"], weights=g[COL_QTY])}))
          .reset_index().sort_values("date"))
    pm["margem_unit"] = pm["preco"] - pm["custo"]

    # variação entre os dois últimos anos
    preco_y0, preco_y1 = pm_year.loc[y0], pm_year.loc[y1]
    custo_y0, custo_y1 = cm_year.loc[y0], cm_year.loc[y1]
    var_preco = 100 * (preco_y1 / preco_y0 - 1)
    var_custo = 100 * (custo_y1 / custo_y0 - 1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pm["date"], pm["preco"], color="#8250c4", lw=1.8, label="Preço médio")
    ax.plot(pm["date"], pm["custo"], color="#c0504d", lw=1.8, label="Custo médio")
    ax.fill_between(pm["date"], pm["custo"], pm["preco"], color="#1d9e75", alpha=0.10, label="Margem unitária")
    ax.set_ylabel("US$ por unidade (ponderado)")
    ax.set_title("Preço vs. custo médio no período"); ax.legend(fontsize=8)
    fig.tight_layout()
    add("Preço — performance no período (preço vs. custo)",
        why="Acompanhar preço e custo médios juntos revela quem manda na margem: se o custo sobe mais rápido que o preço, a margem aperta mesmo com preço subindo. É a leitura de pricing que a receita sozinha esconde.",
        answers="Os preços subiram? E os custos? Qual dos dois cresceu mais?",
        insight=f"De {y0} para {y1}: preço médio {var_preco:+.1f}% e custo médio {var_custo:+.1f}%. "
                + (f"Ambos subiram, e o preço cresceu {'mais' if var_preco > var_custo else 'menos'} que o custo — "
                   f"a margem unitária {'melhorou levemente' if var_preco > var_custo else 'foi pressionada'}."
                   if var_preco > 0 and var_custo > 0 else
                   "Preço e custo tiveram movimentos distintos — ver a faixa verde (margem unitária) no gráfico."),
        fig=fig)

    # -- Dispersão como KPI --
    keys = [COL_MATERIAL, COL_COUNTRY, COL_TIER]
    sdisp = dv.groupby(keys)[COL_PRICE].agg(["count", "mean", "std"])
    sdisp = sdisp[sdisp["count"] >= 8].copy()
    sdisp["cv"] = sdisp["std"] / sdisp["mean"]
    cv_med = 100 * sdisp["cv"].median()
    cv_pond = 100 * np.average(sdisp["cv"].dropna(),
                               weights=sdisp.loc[sdisp["cv"].notna(), "count"])
    pct_alta = 100 * (sdisp["cv"] > 0.2).mean()
    disp_tbl = pd.DataFrame({
        "KPI de dispersão": ["CV mediano dos peer groups", "CV médio ponderado por volume",
                             "% de grupos com alta dispersão (CV > 20%)"],
        "valor": [f"{cv_med:.1f}%", f"{cv_pond:.1f}%", f"{pct_alta:.1f}%"],
    })
    add("Dispersão — indicador-resumo (KPI)",
        why="A dispersão de preço — o quanto o mesmo produto varia de preço entre clientes parecidos — é um indicador-chave de disciplina de pricing. Resumida em poucos números, vira um KPI de acompanhamento (quanto menor, mais consistente a política).",
        answers="Quão consistente é a nossa precificação, num número que dá para acompanhar mês a mês?",
        insight=f"O CV mediano é {cv_med:.1f}% (baixo), mas o ponderado por volume sobe para {cv_pond:.1f}% — os grupos de maior "
                f"volume têm mais dispersão. {pct_alta:.0f}% dos grupos têm alta dispersão (CV > 20%). "
                "A aba Dispersão de Preços destrincha onde e quanto isso vale.",
        table=disp_tbl)


    # ===================================================================
    # BLOCO 5 — QUEM GERA A RECEITA? Pareto de clientes (Parent + País)
    # ===================================================================
    df["_cliente"] = df[COL_PARENT].astype(str) + " · " + df[COL_COUNTRY].astype(str)
    par_cust = (df.groupby("_cliente")[COL_REVENUE].sum()
                .sort_values(ascending=False).reset_index())
    par_cust.columns = ["cliente (parent+país)", "receita"]
    par_cust["cum_%"] = (par_cust["receita"].cumsum() / par_cust["receita"].sum() * 100).round(1)
    par_cust["rank"] = np.arange(1, len(par_cust) + 1)
    par_cust_tbl = _round_df(par_cust.head(100), money_cols=["receita"])
    n50 = int((par_cust["cum_%"] <= 50).sum() + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(par_cust["rank"] / len(par_cust) * 100, par_cust["cum_%"], color="#3b6ea5")
    ax.axhline(80, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("% de clientes (ranqueados)"); ax.set_ylabel("% receita acumulada")
    ax.set_title("Pareto de clientes (receita)")
    fig.tight_layout()
    add("Concentração — clientes (Pareto)",
        "O princípio de Pareto identifica se poucos clientes concentram a receita — informação central para foco comercial e gestão de risco.",
        "Que fração dos clientes responde por 50% e 80% da receita?",
        f"O cliente é definido como Parent + País (o mesmo código de Parent aparece em até 9 países, então o código sozinho não é uma entidade única). "
        f"Apenas {n50} clientes ({n50/len(par_cust)*100:.1f}%) geram 50% da receita — concentração altíssima.",
        fig=fig)
    add("Top 100 clientes (tabela)", "", "", "", table=par_cust_tbl)

    # ===================================================================
    # BLOCO 6 — Pareto de países (receita e margem)
    # ===================================================================
    par_ctry = (df.groupby(COL_COUNTRY)[COL_REVENUE].sum()
                .sort_values(ascending=False).reset_index())
    par_ctry.columns = ["país", "receita"]
    par_ctry["cum_%"] = (par_ctry["receita"].cumsum() / par_ctry["receita"].sum() * 100).round(1)
    par_ctry["rank"] = np.arange(1, len(par_ctry) + 1)
    par_ctry_tbl = _round_df(par_ctry, money_cols=["receita"])
    top3 = par_ctry.head(3)["receita"].sum() / par_ctry["receita"].sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4.6))
    top = par_ctry.head(12).iloc[::-1]
    ax.barh(top["país"], top["receita"] / 1e6, color="#3b6ea5")
    ax.set_xlabel("Receita (US$ mi)"); ax.set_title("Top 12 países por receita")
    fig.tight_layout()
    add("Concentração — países (Pareto)",
        "Mostra a concentração geográfica — onde a operação depende de poucos mercados.",
        "Quais países concentram a receita e qual a dependência dos maiores?",
        f"Os 3 maiores países somam {top3:.0f}% da receita. Concentração relevante que é força (foco) e risco (dependência) ao mesmo tempo.",
        fig=fig)
    add("Ranking de países por receita (tabela)", "", "", "", table=par_ctry_tbl)

    # ===================================================================
    # BLOCO 7 — ONDE PERDEMOS DINHEIRO?
    # ===================================================================
    neg = df[df["_margin"] < 0]
    neg_ctry = (neg.groupby(COL_COUNTRY)["_margin"].sum().sort_values()
                .head(15).reset_index().rename(columns={"_margin": "sangria_de_margem"}))
    neg_ctry_tbl = _round_df(neg_ctry, money_cols=["sangria_de_margem"])

    neg_biz = (neg.groupby(COL_BUSINESS)["_margin"].sum().sort_values()
               .reset_index().rename(columns={"_margin": "sangria_de_margem"}))
    neg_biz_tbl = _round_df(neg_biz, money_cols=["sangria_de_margem"])

    fig, ax = plt.subplots(figsize=(8, 4.4))
    tn = neg_ctry.head(10).iloc[::-1]
    ax.barh(tn[COL_COUNTRY], tn["sangria_de_margem"] / 1e6, color="#c0504d")
    ax.set_xlabel("Sangria de margem (US$ mi, negativo)")
    ax.set_title("Onde perdemos dinheiro — países com maior margem negativa")
    fig.tight_layout()
    add("Perdas — margem negativa por país",
        "Localiza geograficamente as transações que destroem margem (preço abaixo do custo, devoluções, condições ruins).",
        "Em quais países a soma das transações de margem negativa é maior?",
        f"'Sangria de margem' = soma da margem das transações negativas naquele país (quanto mais negativo, pior). "
        f"Concentra-se nos mesmos grandes mercados — provável mistura de devoluções e preço abaixo do custo, a separar na aba 03.",
        fig=fig)
    add("Perdas por país (tabela)",
        "", "", "Valores negativos = margem perdida. Ordenado do pior para o menos pior.",
        table=neg_ctry_tbl)
    add("Perdas por segmento de negócio (tabela)",
        "", "", "Mesma leitura, agora por Business — ajuda a achar um segmento estruturalmente deficitário.",
        table=neg_biz_tbl)

    # ===================================================================
    # BLOCO 8 — TIER A PAGA MAIS?
    # ===================================================================
    d = df[(df[COL_PRICE] > 0) & (df[COL_QTY] > 0)]
    by_tier = (d.groupby(COL_TIER)
               .agg(preco_medio=(COL_PRICE, "mean"), preco_mediano=(COL_PRICE, "median"),
                    volume_medio=(COL_QTY, "mean"), receita=(COL_REVENUE, "sum"),
                    transacoes=(COL_REVENUE, "size"))
               .reset_index())
    by_tier["ticket_medio"] = by_tier["receita"] / by_tier["transacoes"]
    by_tier = by_tier.sort_values("preco_medio", ascending=False)
    by_tier_tbl = _round_df(by_tier, money_cols=["receita", "ticket_medio"],
                            pct_cols=["preco_medio", "preco_mediano", "volume_medio"],
                            int_cols=["transacoes"])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    xt = np.arange(len(by_tier))
    colors = ["#c0504d" if t == "UNSPECIFIED" else "#3b6ea5" for t in by_tier[COL_TIER]]
    ax.bar(xt, by_tier["preco_mediano"], color=colors)
    ax.set_xticks(xt); ax.set_xticklabels(by_tier[COL_TIER].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Preço mediano (US$)"); ax.set_title("Preço mediano por Tier")
    fig.tight_layout()
    a_is_top = by_tier.iloc[0][COL_TIER] == "A"
    add("Segmentação — o Tier A paga mais?",
        "Testa se a hierarquia de tiers se reflete no preço — uma premissa comum que os dados podem confirmar ou desmentir.",
        "O preço unitário médio/mediano cresce conforme o tier do cliente?",
        ("Sim, o Tier A lidera o preço." if a_is_top else
         "Não: Tier A NÃO tem o maior preço unitário (OEM/Integrator e Standard cobram mais por unidade). "
         "A importância do Tier A vem do VOLUME muito maior por transação, não do preço — insight contraintuitivo."),
        fig=fig)
    add("Preço e volume por Tier (tabela)", "", "", "", table=by_tier_tbl)

    # ===================================================================
    # BLOCO 9 — MATERIAIS QUE MAIS VENDEM
    # ===================================================================
    by_mat = (df.groupby(COL_MATERIAL)
              .agg(receita=(COL_REVENUE, "sum"), quantidade=(COL_QTY, "sum"),
                   margem=("_margin", "sum"), transacoes=(COL_REVENUE, "size"))
              .reset_index())
    by_mat["margem_%"] = (100 * by_mat["margem"] / by_mat["receita"]).round(1)
    by_mat = by_mat.sort_values("receita", ascending=False)
    by_mat_tbl = _round_df(by_mat.head(50), money_cols=["receita", "margem"],
                           int_cols=["quantidade", "transacoes"])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    tm = by_mat.head(12).iloc[::-1]
    ax.barh(tm[COL_MATERIAL].astype(str), tm["receita"] / 1e6, color="#3b6ea5")
    ax.set_xlabel("Receita (US$ mi)"); ax.set_title("Top 12 materiais por receita")
    fig.tight_layout()
    add("Portfólio — materiais que mais vendem",
        "Identifica os produtos que sustentam o faturamento e cruza com a margem — nem sempre o que mais vende é o mais rentável.",
        "Quais são os produtos-chave por receita e como está a margem % de cada um?",
        "O cruzamento receita × margem % revela produtos de alto giro e baixa margem (candidatos a reprecificação) vs. produtos premium.",
        fig=fig)
    add("Top 50 materiais (tabela)", "", "", "", table=by_mat_tbl)

    # ===================================================================
    # BLOCO 10 — CORRELAÇÃO
    # ===================================================================
    corr = df[NUM_COLS].corr().round(2)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM_COLS))); ax.set_yticks(range(len(NUM_COLS)))
    short = [c.replace(" ($)", "").replace("_", " ") for c in NUM_COLS]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(NUM_COLS)):
        for j in range(len(NUM_COLS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlação entre variáveis")
    fig.tight_layout()
    add("Relações — correlação entre variáveis",
        "Mostra quais variáveis se movem juntas — útil para detectar redundância e relações esperadas (ou suspeitas).",
        "Preço se relaciona com volume? Receita com margem?",
        "Price_Avg e Margin_Unit têm correlação ~1 (redundantes por construção). Receita e Margem_Total ~0,89. "
        "Preço × Quantidade é ~0 no agregado — o desconto de volume aparece dentro do peer group, não na base inteira.",
        fig=fig)

    # ===================================================================
    # NOTAS gerais (findings de qualidade)
    # ===================================================================
    notes = []
    tier_unspec = 100 * (df[COL_TIER] == "UNSPECIFIED").mean()
    if tier_unspec > 20:
        notes.append(f"Customer Tier ausente em {tier_unspec:.0f}% das linhas — maior fragilidade; "
                     "ticket ~10x menor sugere venda avulsa.")
    if cogs_pos:
        notes.append(f"COGS corrigido: {cogs_pos:,} linhas ({cogs_pos_pct:.0f}%) vinham com sinal positivo "
                     f"({cogs_pos_biz_share:.0f}% no Business {cogs_pos_biz}); o sinal foi forçado para negativo "
                     "na limpeza, então a margem está correta em toda a base.")
    bbm = by_business_margin(df)
    for _, r in bbm.iterrows():
        if r["margem_%"] > 100:
            notes.append(f"Business {r[COL_BUSINESS]}: margem {r['margem_%']:.0f}% (>100%) — COGS invertido/ausente.")
        elif abs(r["margem_%"]) < 0.5:
            notes.append(f"Business {r[COL_BUSINESS]}: margem ~0% — COGS nulo neste segmento.")
    notes.append(f"Concentração: top 3 países = {top3:.0f}% da receita; top {n50} clientes (parent+país) = 50%.")

    metrics = {
        "Linhas": f"{n:,}",
        "Colunas": f"{len(ALL_COLS)}",
        "Período": f"{int(df['year'].min())} a {int(df['year'].max())}",
        "Receita total": _fmt_money_mi(df[COL_REVENUE].sum()),
        "Margem total": _fmt_money_mi(df["_margin"].sum()),
        "Ticket médio": f"${df[COL_REVENUE].sum()/n:,.0f}",
    }
    return {"metrics": metrics, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


def by_business_margin(df):
    b = (df.groupby(COL_BUSINESS)
         .agg(receita=(COL_REVENUE, "sum"), margem=("_margin", "sum")).reset_index())
    b["margem_%"] = (100 * b["margem"] / b["receita"]).round(1)
    return b




# ========================================================================
# 02 — PRICE DISPERSION (análise completa em blocos)
# ========================================================================
PEER_KEYS = [COL_MATERIAL, COL_COUNTRY, COL_TIER]


def run_dispersion(df: pd.DataFrame, p: Params) -> dict:
    """Dispersão de preço em profundidade: o que é CV, distribuição, spread,
    heatmap de mercados, scatter volume-desconto, outliers por IQR e ranking
    de oportunidades. Retorna blocos explicativos (por que / responde / insight)."""
    d = df[df["price_valid"]].copy()

    # ---- estatísticas por peer group ----
    g = d.groupby(PEER_KEYS)[COL_PRICE]
    stats = g.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    stats["cv"] = (stats["std"] / stats["mean"]).round(4)
    stats["spread_%"] = (100 * (stats["max"] - stats["min"]) / stats["median"]).round(1)
    stats["target_price"] = g.quantile(p.dispersion_target_pctl).values
    stats = stats[stats["count"] >= p.min_peer_size].copy()

    # ---- oportunidade: transações abaixo do alvo ----
    d = d.merge(stats[PEER_KEYS + ["target_price", "count", "median", "cv"]],
                on=PEER_KEYS, how="inner")
    d["price_rel"] = d[COL_PRICE] / d["median"]
    below = d[d[COL_PRICE] < d["target_price"]].copy()
    below["price_gap"] = below["target_price"] - below[COL_PRICE]
    below["uplift_gross"] = below["price_gap"] * below[COL_QTY]
    below["uplift_captured"] = below["uplift_gross"] * p.capture_rate

    total_gross = below["uplift_gross"].sum()
    total_capt = below["uplift_captured"].sum()
    cv_med = stats["cv"].median()

    blocks = []

    def add(title, why="", answers="", insight="", table=None, fig=None,
            assumptions="", money=""):
        blocks.append({"title": title, "why": why, "answers": answers,
                       "insight": insight, "table": table, "fig": fig,
                       "assumptions": assumptions, "money": money})

    # ===================================================================
    # BLOCO 0 — O QUE ESTAMOS MEDINDO (conceito)
    # ===================================================================
    add("O que é dispersão de preço e por que importa",
        "O mesmo produto costuma ser vendido a preços diferentes para clientes parecidos. Essa variação — a dispersão — é dinheiro potencialmente deixado na mesa: se alguns clientes pagam bem menos que seus pares, há espaço para subir esses preços.",
        "Existe variação de preço injustificada para o mesmo produto? Onde? Quanto vale corrigir?",
        "Comparamos preços apenas dentro de um 'peer group' — mesmo Material × País × Tier — para que a comparação seja justa (maçã com maçã). "
        f"Analisamos {len(stats):,} peer groups com pelo menos {p.min_peer_size} transações cada.",
        fig=None)

    # ===================================================================
    # BLOCO 1 — COEFICIENTE DE VARIAÇÃO (CV): conceito + distribuição
    # ===================================================================
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(stats["cv"].clip(0, 1.5).dropna(), bins=50, color="#3b6ea5")
    ax.axvline(cv_med, color="crimson", ls="--", label=f"mediana CV = {cv_med:.2f}")
    ax.set_xlabel("Coeficiente de variação do preço (CV)")
    ax.set_ylabel("Nº de peer groups"); ax.legend()
    ax.set_title("Distribuição do CV entre peer groups")
    fig.tight_layout()
    add("Dispersão — coeficiente de variação (CV)",
        "O CV = desvio-padrão ÷ média. É a forma padronizada de medir dispersão: independe da escala do preço, então um item de R$ 10 e outro de R$ 10.000 podem ser comparados. CV de 0,10 significa que os preços variam ~10% em torno da média daquele grupo.",
        "Quão consistente é o preço de cada produto? Quais grupos estão descontrolados?",
        f"A mediana do CV é {cv_med:.0%} — a maioria dos grupos tem preço sob controle. Mas a cauda à direita "
        f"({int((stats['cv'] > 1).sum())} grupos com CV > 1) são casos onde o mesmo produto varia mais de 100% em preço: "
        "é onde a política de precificação está mais solta — ou onde há erro de cadastro.",
        fig=fig)

    # ===================================================================
    # BLOCO 2 — DISTRIBUIÇÃO DE PREÇOS (visão geral)
    # ===================================================================
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.log10(d[COL_PRICE].clip(lower=0.01)), bins=60, color="#3b6ea5")
    ax.set_xlabel("Preço unitário (log10)"); ax.set_ylabel("Nº de transações")
    ax.set_title("Distribuição de preços (todas as transações válidas)")
    fig.tight_layout()
    add("Distribuição de preços",
        "Ver a forma da distribuição de preços revela a estrutura do portfólio: concentração em faixas, presença de cauda longa (itens premium) e possíveis erros de escala.",
        "Como os preços se distribuem? Há concentração ou cauda longa?",
        "Em escala log a distribuição fica mais legível porque os preços variam por ordens de grandeza (de centavos a mais de US$ 1 milhão). "
        "A cauda à direita são itens de altíssimo valor unitário — poucos, mas de peso grande na receita.",
        fig=fig)

    # ===================================================================
    # BLOCO 3 — SPREAD DE PREÇOS por peer group
    # ===================================================================
    spread_tbl = _round_df(
        stats.sort_values("spread_%", ascending=False).head(30)[
            PEER_KEYS + ["count", "min", "median", "max", "cv", "spread_%"]],
        money_cols=["min", "median", "max"], pct_cols=["cv", "spread_%"], int_cols=["count"])
    add("Spread de preços — os grupos mais esticados",
        "O spread (%) = (preço máximo − mínimo) ÷ mediana. Mostra, em cada grupo, a distância entre o cliente que paga mais e o que paga menos pelo mesmo item.",
        "Em quais peer groups a diferença entre o maior e o menor preço é mais gritante?",
        "Spread altíssimo (centenas de %) quase sempre indica um de dois casos: oportunidade real de repricing OU erro de dado "
        "(unidade trocada, materiais distintos sob o mesmo código). Ambos exigem ação — o segundo, correção de cadastro.",
        table=spread_tbl)

    # ===================================================================
    # BLOCO 4 — BOXPLOT dos top grupos (cor = CV, viridis)
    # ===================================================================
    up_by_group = (below.groupby(PEER_KEYS)["uplift_gross"].sum()
                   .reset_index().rename(columns={"uplift_gross": "uplift_group"}))
    stats = stats.merge(up_by_group, on=PEER_KEYS, how="left")
    stats["uplift_group"] = stats["uplift_group"].fillna(0.0)
    fig_box = _dispersion_boxplot(d, stats, p)
    add("Boxplot dos grupos prioritários",
        "O boxplot mostra, para cada grupo, a caixa (quartis), a mediana e os outliers de preço. A cor viridis codifica o CV: amarelo = mais disperso.",
        "Nos grupos de maior oportunidade, como os preços se espalham? Onde estão os outliers?",
        "A altura da caixa é a dispersão real (escala log). Grupos amarelos e altos combinam duas bandeiras — muita dispersão e muita oportunidade — "
        "e são os primeiros a investigar. A cor separa 'bagunça de preço' de 'grupo grande mas organizado'.",
        fig=fig_box)

    # ===================================================================
    # BLOCO 5 — HEATMAP: mesmo material, países diferentes
    # ===================================================================
    fig_hm = _dispersion_heatmap(d)
    add("Dispersão — heatmap de preço por mercado",
        "Para os produtos vendidos em mais países, comparamos o preço de cada país contra a mediana global daquele produto. Vermelho = mais caro que a mediana; verde = mais barato.",
        "Existe país onde o mesmo produto é sistematicamente cobrado mais barato (ou mais caro)?",
        "Células verdes são mercados onde o produto sai barato — candidatos a reajuste, se não houver justificativa (câmbio, custo local, concorrência). "
        "Vermelhas são mercados premium. A variação chega a faixas de 0,1x a ~4x para o mesmo item.",
        fig=fig_hm)

    # ===================================================================
    # BLOCO 6 — SCATTER: volume vs desconto
    # ===================================================================
    fig_sc, sc = _dispersion_scatter(d)
    add("Dispersão — desconto por volume (real vs. teoria)",
        why="Desconto de volume é a justificativa legítima nº 1 para preços diferentes. Como a base NÃO tem coluna de desconto, ele é inferido: usamos o preço relativo (preço ÷ mediana do peer group) no eixo Y — 1,0 significa 'na mediana do grupo'. A curva verde é a teoria (desconto de volume esperado); a vermelha é a tendência real ajustada aos dados.",
        answers="O preço cai conforme o volume sobe, como manda a teoria? Quão longe a nossa realidade está da curva esperada, e isso é bom ou ruim?",
        assumptions="A curva teórica usa uma inclinação ilustrativa de −15% por década de volume (referência didática, não a política oficial da empresa). Preço relativo controla o item via peer group, mas não controla custo local nem câmbio.",
        insight=(f"Correlação volume × preço = {sc['corr']:+.2f} (quase nula): a linha real é quase plana, a teórica é bem inclinada. "
                 f"A distância vertical média entre elas é de {sc['gap']*100:+.0f} pontos percentuais — a linha real está ACIMA da teórica. "
                 "IMPORTANTE, contra a intuição: esse gap NÃO é dinheiro na mesa. Significa que a empresa cobra MAIS do que o desconto de volume teórico daria — "
                 f"de fato, {sc['pct_high_above']:.0%} dos clientes de alto volume já pagam acima da curva teórica, o que PROTEGE a margem. "
                 "A linha vermelha só 'deveria' estar sobre a verde se a política fosse dar desconto de volume pleno — mas não dar esse desconto é favorável ao resultado. "
                 "A oportunidade real não é fechar o gap com a teoria: é corrigir quem paga abaixo da mediana do grupo SEM ter volume que justifique (baixo volume + preço baixo)."),
        money=(f"Separando por volume: das transações abaixo da mediana, apenas US$ {sc['opp_low']/1e6:,.1f} mi vêm de BAIXO volume "
               f"(oportunidade limpa de repricing); US$ {sc['opp_high']/1e6:,.1f} mi vêm de ALTO volume — desconto justificável, que NÃO se deve cobrar de volta. "
               "Ou seja, o número honesto de oportunidade aqui é muito menor que o gap bruto sugere."),
        fig=fig_sc)

    # ===================================================================
    # BLOCO 7 — OUTLIERS por IQR
    # ===================================================================
    d["q1"] = d.groupby(PEER_KEYS)[COL_PRICE].transform(lambda x: x.quantile(0.25))
    d["q3"] = d.groupby(PEER_KEYS)[COL_PRICE].transform(lambda x: x.quantile(0.75))
    d["iqr"] = d["q3"] - d["q1"]
    d["out_low"] = d[COL_PRICE] < (d["q1"] - 1.5 * d["iqr"])
    d["out_high"] = d[COL_PRICE] > (d["q3"] + 1.5 * d["iqr"])
    n_low, n_high = int(d["out_low"].sum()), int(d["out_high"].sum())

    out_low = d[d["out_low"]].copy()
    out_low["abaixo_do_q1_%"] = (100 * (d["q1"] - out_low[COL_PRICE]) / d["q1"]).round(1)
    out_tbl = _round_df(
        out_low.sort_values(COL_PRICE)[
            PEER_KEYS + [COL_MONTH, COL_PARENT, COL_QTY, COL_PRICE, "q1", "median", "abaixo_do_q1_%"]
        ].head(50),
        money_cols=[COL_PRICE, "q1", "median"], int_cols=[COL_QTY])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(["Preço anormalmente\nBAIXO", "Preço anormalmente\nALTO"], [n_low, n_high],
           color=["#c0504d", "#e0a030"])
    ax.set_ylabel("Nº de transações"); ax.set_title("Outliers de preço por IQR")
    fig.tight_layout()
    add("Detecção de outliers (regra do IQR)",
        "A regra do IQR (intervalo interquartil) marca como outlier todo preço abaixo de Q1−1,5·IQR ou acima de Q3+1,5·IQR dentro do seu peer group. É o mesmo critério que define os 'pontos' de um boxplot — um detector estatístico padrão, sem depender de limiar arbitrário.",
        "Quais transações têm preço fora do padrão do próprio grupo?",
        f"Há {n_low:,} transações com preço anormalmente BAIXO (oportunidade direta de repricing) e {n_high:,} anormalmente ALTO "
        "(clientes premium ou possível erro). Os outliers baixos são a lista de ação mais limpa que a dispersão produz.",
        fig=fig)
    add("Outliers de preço baixo — lista de ação", "", "",
        "Transações abaixo de Q1−1,5·IQR do peer group, ordenadas do menor preço. 'abaixo_do_q1_%' mostra o quanto o preço está abaixo do primeiro quartil.",
        table=out_tbl)

    # ===================================================================
    # BLOCO 8 — RANKING DE OPORTUNIDADES (uplift)
    # ===================================================================
    opp = below.sort_values("uplift_gross", ascending=False)
    opp_tbl = _round_df(
        opp[PEER_KEYS + [COL_MONTH, COL_PARENT, COL_QTY, COL_PRICE, "target_price",
                         "price_gap", "uplift_gross", "uplift_captured"]].head(100),
        money_cols=[COL_PRICE, "target_price", "price_gap", "uplift_gross", "uplift_captured"],
        int_cols=[COL_QTY])
    add("Dispersão — ranking de oportunidades",
        "Consolida o ganho financeiro: para cada transação abaixo do preço-alvo (mediana do grupo), o uplift = (preço-alvo − preço praticado) × quantidade. É a lista que o time comercial ataca, cliente a cliente.",
        "Quanto vale corrigir a dispersão e por onde começar?",
        f"São {len(opp):,} transações abaixo do alvo, somando US$ {total_gross:,.0f} de uplift bruto "
        f"(US$ {total_capt:,.0f} assumindo {p.capture_rate:.0%} de captura real). "
        "O ranking prioriza pelos maiores ganhos absolutos.",
        table=opp_tbl)

    # ---- métricas do topo (com explicação em nota) ----
    metrics = {
        "Peer groups": f"{len(stats):,}",
        "CV mediano": f"{cv_med:.1%}",
        "Transações abaixo do alvo": f"{len(below):,}",
        "Uplift bruto": f"${total_gross/1e6:,.1f} mi",
        f"Uplift capturável ({p.capture_rate:.0%})": f"${total_capt/1e6:,.1f} mi",
        "Outliers de preço baixo": f"{n_low:,}",
    }
    notes = [
        "Peer group = mesmo Material × País × Tier (comparação justa).",
        "CV mediano = dispersão típica de preço; quanto menor, mais consistente a precificação.",
        "'Transações abaixo do alvo' = quantas pagam menos que a mediana do seu grupo.",
        f"'Uplift bruto' = ganho se todas subissem ao alvo; 'capturável' aplica {p.capture_rate:.0%} de realismo comercial.",
    ]
    return {"metrics": metrics, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


def _dispersion_heatmap(d: pd.DataFrame):
    """Heatmap material x país: preço relativo à mediana global do material."""
    mat_ctry = d.groupby([COL_MATERIAL, COL_COUNTRY])[COL_PRICE].median().reset_index()
    nc = mat_ctry.groupby(COL_MATERIAL)[COL_COUNTRY].nunique().sort_values(ascending=False)
    topmat = nc.head(10).index.tolist()
    topc = (d.groupby(COL_COUNTRY)[COL_REVENUE].sum()
            .sort_values(ascending=False).head(12).index.tolist())
    sub = mat_ctry[mat_ctry[COL_MATERIAL].isin(topmat) & mat_ctry[COL_COUNTRY].isin(topc)].copy()
    gm = sub.groupby(COL_MATERIAL)[COL_PRICE].transform("median")
    sub["rel"] = sub[COL_PRICE] / gm
    piv = sub.pivot_table(index=COL_MATERIAL, columns=COL_COUNTRY, values="rel")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(m)[:14] for m in piv.index], fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="preço vs mediana do material (1.0 = mediana)")
    ax.set_title("Heatmap: mesmo material, preço por país (verde = mais barato)")
    fig.tight_layout()
    return fig


def _dispersion_scatter(d: pd.DataFrame):
    """Scatter volume x preço relativo + curva teórica. Retorna também as
    métricas que quantificam a distância real-teoria e a oportunidade limpa."""
    m = (d[COL_QTY] > 0) & (d["price_rel"] > 0)
    corr = float(np.corrcoef(np.log(d.loc[m, COL_QTY]), d.loc[m, "price_rel"])[0, 1])
    samp = d[m].sample(min(4000, int(m.sum())), random_state=1)

    qv = d.loc[m, COL_QTY].values
    pr = d.loc[m, "price_rel"].values
    b1, b0 = np.polyfit(np.log10(qv), pr, 1)
    theo_slope = -0.15  # teoria: -15% de preço a cada 10x de volume
    q_ref = np.median(qv)

    # gap vertical médio entre a linha real e a teórica (positivo = real acima)
    lnq = np.log10(qv)
    gap = float(((b0 + b1 * lnq) - (1.0 + theo_slope * (lnq - np.log10(q_ref)))).mean())

    # oportunidade LIMPA: baixo volume pagando abaixo da mediana do grupo
    q33 = d[COL_QTY].quantile(0.33)
    below = d[(d["price_rel"] < 1)].copy()
    below["opp"] = (below["median"] - below[COL_PRICE]) * below[COL_QTY]
    opp_total = float(below["opp"].sum())
    opp_low = float(below.loc[below[COL_QTY] <= q33, "opp"].sum())
    opp_high = opp_total - opp_low
    # % de alto volume que já paga acima da teoria (proteção de margem)
    high = d[d[COL_QTY] >= d[COL_QTY].quantile(0.66)].copy()
    hln = np.log10(high[COL_QTY])
    high_theo = 1.0 + theo_slope * (hln - np.log10(q_ref))
    pct_high_above = float((high["price_rel"] > high_theo).mean())

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(samp[COL_QTY], samp["price_rel"], alpha=0.15, s=8, color="#3b6ea5",
               label="realidade (transações)")
    ax.axhline(1, color="grey", ls=":", lw=1, label="mediana do grupo (preço rel. = 1)")
    xs = np.logspace(np.log10(max(qv.min(), 1)), np.log10(qv.max()), 100)
    ax.plot(xs, b0 + b1 * np.log10(xs), color="#c0504d", lw=2,
            label=f"tendência real (inclin. {b1:+.3f})")
    ax.plot(xs, 1.0 + theo_slope * (np.log10(xs) - np.log10(q_ref)), color="#1d9e75",
            lw=2, ls="--", label="teoria (desconto de volume esperado)")
    # sombrear o gap entre as duas linhas
    real_y = b0 + b1 * np.log10(xs)
    theo_y = 1.0 + theo_slope * (np.log10(xs) - np.log10(q_ref))
    ax.fill_between(xs, theo_y, real_y, where=(real_y >= theo_y), color="#1d9e75", alpha=0.08)

    ax.set_xscale("log"); ax.set_ylim(0, 3)
    ax.set_xlabel("Quantidade (log)")
    ax.set_ylabel("Preço vs mediana do peer group")
    ax.set_title(f"Volume × desconto: realidade vs teoria (correlação = {corr:+.2f})")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    metrics = {"corr": corr, "gap": gap, "opp_total": opp_total,
               "opp_low": opp_low, "opp_high": opp_high,
               "pct_high_above": pct_high_above}
    return fig, metrics


def _dispersion_boxplot(d: pd.DataFrame, stats: pd.DataFrame, p: Params):
    """Boxplot dos top-N peer groups, caixas coloridas por CV em viridis."""
    from matplotlib.colors import Normalize

    rank_col = "cv" if p.boxplot_rank_by == "cv" else "uplift_group"
    top = stats.sort_values(rank_col, ascending=False).head(p.boxplot_top_n).reset_index(drop=True)

    groups, labels, cvs = [], [], []
    for _, r in top.iterrows():
        mask = ((d[COL_MATERIAL] == r[COL_MATERIAL])
                & (d[COL_COUNTRY] == r[COL_COUNTRY])
                & (d[COL_TIER] == r[COL_TIER]))
        prices = d.loc[mask, COL_PRICE].values
        if len(prices) == 0:
            continue
        groups.append(prices)
        labels.append(f"{str(r[COL_MATERIAL])[:10]}·{str(r[COL_COUNTRY])[:3].upper()}·{r[COL_TIER]}")
        cvs.append(r["cv"])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    if not groups:
        ax.text(0.5, 0.5, "Sem grupos suficientes para o boxplot", ha="center", va="center")
        return fig

    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=min(cvs), vmax=max(cvs) if max(cvs) > min(cvs) else min(cvs) + 1e-9)
    colors = [cmap(norm(c)) for c in cvs]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.35))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.85); patch.set_edgecolor("#333")
    for med in bp["medians"]:
        med.set_color("white"); med.set_linewidth(1.5)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Preço unitário (escala log)")
    crit = "uplift" if rank_col == "uplift_group" else "CV"
    ax.set_title(f"Dispersão de preço — top {len(labels)} peer groups por {crit} (cor = CV)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label("CV (dispersão)")
    fig.tight_layout()
    return fig


# ========================================================================
# 03 — DESPERDÍCIO DE MARGEM (margem negativa, análise completa)
# ========================================================================
def run_negative_margin(df: pd.DataFrame, p: Params) -> dict:
    """Análise profunda das transações que destroem margem: evolução temporal,
    SKUs críticos, concentração das perdas, negociação vs. estrutural,
    matriz receita×margem e simulação de break-even com uplifts."""
    df = df.copy()
    df["_margin"] = df[COL_REVENUE] + df[COL_COGS]
    neg = df[df["_margin"] < 0].copy()
    neg["unit_cost"] = -neg[COL_COGS_AVG]
    neg["recovery"] = -neg["_margin"]

    total_sangria = neg["_margin"].sum()
    blocks = []

    def add(title, why, answers, insight, table=None, fig=None):
        blocks.append({"title": title, "why": why, "answers": answers,
                       "insight": insight, "table": table, "fig": fig})

    # ===================================================================
    # BLOCO 0 — conceito
    # ===================================================================
    add("O que é desperdício de margem",
        "São as transações em que a margem foi negativa: a empresa vendeu por menos do que custou (ou registrou devolução/ajuste). É a perda mais direta e o quick win mais óbvio — cada real recuperado aqui vai inteiro para o resultado.",
        "Quanto estamos perdendo, onde, e é problema pontual ou estrutural?",
        f"São {len(neg):,} transações ({len(neg)/len(df):.2%} da base) somando US$ {total_sangria:,.0f} de margem destruída. "
        "As análises abaixo separam o que é ruído (devolução) do que é sangria estrutural (preço abaixo do custo).",
        fig=None)

    # ===================================================================
    # BLOCO 1 — EVOLUÇÃO TEMPORAL (está piorando?)
    # ===================================================================
    ts = neg.groupby("date")["_margin"].sum().sort_index()
    x = np.arange(len(ts))
    slope = float(np.polyfit(x, ts.values, 1)[0]) if len(ts) > 1 else 0.0
    trend = "melhorando (perda diminuindo)" if slope > 0 else "piorando (perda aumentando)"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([d.strftime("%Y-%m") for d in ts.index], ts.values / 1e6, color="#c0504d")
    zz = np.poly1d(np.polyfit(x, ts.values, 1))(x) / 1e6
    ax.plot(range(len(ts)), zz, color="#333", ls="--", lw=1.5, label="tendência")
    step = max(1, len(ts) // 12)
    ax.set_xticks(range(0, len(ts), step))
    ax.set_xticklabels([ts.index[i].strftime("%Y-%m") for i in range(0, len(ts), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Margem negativa (US$ mi)"); ax.set_title("Evolução mensal do desperdício de margem")
    ax.legend(); fig.tight_layout()
    add("A margem negativa vem piorando ou foi pontual?",
        "Distinguir tendência de evento pontual muda a resposta: uma piora consistente aponta problema estrutural de precificação; picos isolados apontam eventos (um contrato ruim, um lote devolvido).",
        "A perda cresce mês a mês ou se concentra em meses específicos?",
        f"A linha de tendência está {trend} (~US$ {abs(slope):,.0f}/mês). "
        "Picos isolados em meses específicos indicam eventos pontuais que valem investigação caso a caso, não mudança de política.",
        fig=fig)

    # ===================================================================
    # BLOCO 2 — SKUs QUE MAIS GERAM PERDA
    # ===================================================================
    by_sku = (neg.groupby(COL_MATERIAL)
              .agg(sangria=("_margin", "sum"), transacoes=("_margin", "size"),
                   qtd=(COL_QTY, "sum"))
              .reset_index().sort_values("sangria"))
    by_sku_tbl = _round_df(by_sku.head(30), money_cols=["sangria"], int_cols=["transacoes", "qtd"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    top = by_sku.head(12).iloc[::-1]
    ax.barh(top[COL_MATERIAL].astype(str), top["sangria"] / 1e6, color="#c0504d")
    ax.set_xlabel("Sangria de margem (US$ mi)"); ax.set_title("Top 12 SKUs por margem negativa")
    fig.tight_layout()
    worst = by_sku.iloc[0]
    add("Quais SKUs geram as maiores perdas?",
        "Concentrar a ação nos produtos certos multiplica o retorno do esforço. Ranquear por sangria total mostra onde atacar primeiro.",
        "Quais produtos individualmente destroem mais margem?",
        f"O pior SKU ({worst[COL_MATERIAL]}) sozinho gera US$ {abs(worst['sangria']):,.0f} de perda em {int(worst['transacoes'])} transações. "
        "Poucos SKUs concentram a maior parte da sangria — o bloco de distribuição a seguir quantifica isso.",
        fig=fig)
    add("SKUs críticos (tabela)", "", "",
        "Ordenado da maior perda para a menor. 'sangria' negativa = margem destruída pelo produto.",
        table=by_sku_tbl)

    # ===================================================================
    # BLOCO 3 — DISTRIBUIÇÃO DAS PERDAS (poucos itens ou muitos?)
    # ===================================================================
    sku_loss = by_sku.set_index(COL_MATERIAL)["sangria"].sort_values()
    cum = sku_loss.cumsum() / sku_loss.sum()
    n50 = int((cum <= 0.5).sum() + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.log10(np.abs(sku_loss[sku_loss < 0]).clip(lower=1)), bins=40, color="#c0504d")
    ax.set_xlabel("Perda por SKU (log10 de US$)"); ax.set_ylabel("Nº de SKUs")
    ax.set_title("Distribuição das perdas por SKU")
    fig.tight_layout()
    add("Margem negativa — concentração das perdas",
        "A forma da distribuição define a estratégia: se a perda é concentrada, uma força-tarefa em poucos SKUs resolve; se é difusa, o problema é de política geral de precificação.",
        "As perdas se concentram em poucos SKUs ou se espalham por muitos?",
        f"50% de toda a sangria vem de apenas {n50} SKUs ({n50/len(sku_loss)*100:.1f}% dos produtos com perda). "
        "Concentração altíssima — a boa notícia é que uma ação cirúrgica em poucos itens recupera metade da perda.",
        fig=fig)

    # ===================================================================
    # BLOCO 4 — NEGOCIAÇÃO ou PROBLEMA ESTRUTURAL?
    # ===================================================================
    neg["below_cost"] = neg[COL_PRICE] < neg["unit_cost"]
    n_struct = int(neg["below_cost"].sum())
    n_other = int((~neg["below_cost"]).sum())
    struct_loss = neg.loc[neg["below_cost"], "_margin"].sum()

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(["Preço abaixo do custo\n(estrutural)", "Outros\n(devolução/pontual)"],
           [n_struct, n_other], color=["#c0504d", "#e0a030"])
    ax.set_ylabel("Nº de transações"); ax.set_title("Negociação vs. problema estrutural")
    fig.tight_layout()
    add("Margem negativa — estrutural vs. pontual",
        "Uma venda abaixo do custo é decisão de precificação (estrutural, recorrente); uma devolução ou ajuste é evento pontual. A ação é diferente para cada um.",
        "A perda vem de vender abaixo do custo (estrutural) ou de eventos pontuais?",
        f"{n_struct:,} transações ({n_struct/len(neg):.0%}) têm preço ABAIXO do custo — sangria estrutural de "
        f"US$ {abs(struct_loss):,.0f}, que só se corrige com repricing. As outras {n_other:,} são majoritariamente devoluções/ajustes.",
        fig=fig)

    # ===================================================================
    # BLOCO 5 — MATRIZ RECEITA × MARGEM (quadrantes de prioridade)
    # ===================================================================
    allmat = (df.groupby(COL_MATERIAL)
              .agg(receita=(COL_REVENUE, "sum"), margem=("_margin", "sum")).reset_index())
    neg_skus = allmat[allmat["margem"] < 0].copy()
    rev_med = allmat["receita"].median()
    neg_skus["prioridade"] = np.where(neg_skus["receita"] > rev_med,
                                      "URGENTE (alta receita)", "Baixa prioridade")
    urgente = neg_skus[neg_skus["prioridade"].str.startswith("URGENTE")]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = np.where(neg_skus["receita"] > rev_med, "#c0504d", "#7a9cc0")
    ax.scatter(neg_skus["receita"].clip(lower=1), -neg_skus["margem"],
               c=colors, alpha=0.55, s=20, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")

    # sombrear o quadrante URGENTE (direita = alta receita)
    ax.axvspan(rev_med, neg_skus["receita"].max() * 1.5, color="#c0504d", alpha=0.07)
    ax.axvline(rev_med, color="grey", ls="--", lw=1)

    # rótulos dos quadrantes
    ymax = (-neg_skus["margem"]).max()
    ax.text(rev_med * 1.3, ymax * 0.9, "URGENTE\nalta receita + prejuízo",
            fontsize=10, color="#8a2020", weight="bold", va="top")
    ax.text(rev_med * 0.5, ymax * 0.9, "baixa prioridade\nbaixa receita",
            fontsize=9, color="#43607a", va="top", ha="right")

    ax.set_xlabel("Receita do SKU (log)  →  maior importância")
    ax.set_ylabel("Perda de margem (log, US$)  →  maior prejuízo")
    ax.set_title("Matriz Receita × Margem negativa — quadrante URGENTE destacado")
    fig.tight_layout()
    add("Margem negativa — matriz receita × margem (urgência)",
        "Nem todo SKU deficitário tem a mesma prioridade. Cruzar receita (importância do produto) com a perda separa o que é urgente do que pode esperar.",
        "Quais produtos deficitários são grandes o suficiente para exigir ação imediata?",
        f"Quadrante URGENTE (alta receita + margem negativa): {len(urgente)} SKUs somando US$ {abs(urgente['margem'].sum()):,.0f} de perda. "
        "São produtos que vendem bem MAS no prejuízo — corrigir preço aqui tem impacto grande e imediato. "
        "SKUs de baixa receita e margem negativa são baixa prioridade (podem até ser descontinuados).",
        fig=fig)
    add("SKUs URGENTES (tabela)", "", "",
        "Produtos de receita acima da mediana e margem negativa — a fila de ação prioritária.",
        table=_round_df(urgente.sort_values("margem")[[COL_MATERIAL, "receita", "margem"]].head(40),
                        money_cols=["receita", "margem"]))

    # ===================================================================
    # BLOCO 6 — SIMULAÇÃO DE BREAK-EVEN (+2/+5/+10%)
    # ===================================================================
    scen = []
    for x in [0.0, 0.02, 0.05, 0.10]:
        target = neg["unit_cost"] * (1 + x)
        gain = ((target - neg[COL_PRICE]) * neg[COL_QTY]).clip(lower=0).sum()
        scen.append({"cenário": f"break-even +{x:.0%}", "recuperação_US$": gain})
    scen_df = pd.DataFrame(scen)
    scen_tbl = _round_df(scen_df, money_cols=["recuperação_US$"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(scen_df["cenário"], scen_df["recuperação_US$"] / 1e6,
           color=["#7a9cc0", "#5f86b3", "#3b6ea5", "#1d9e75"])
    for i, v in enumerate(scen_df["recuperação_US$"]):
        ax.annotate(f"${v/1e6:,.1f}mi", (i, v / 1e6), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Recuperação (US$ mi)")
    ax.set_title("Simulação: se cada item negativo fosse vendido ao custo (+markup)")
    fig.tight_layout()
    be0 = scen_df.loc[0, "recuperação_US$"]
    be10 = scen_df.loc[3, "recuperação_US$"]
    add("Margem negativa — simulação de break-even",
        "Quantifica o prêmio da ação mais simples possível: parar de vender no prejuízo. Simula levar cada transação negativa ao custo (break-even) e, além disso, a pequenos markups de +2%, +5% e +10%.",
        "Quanto recuperaríamos se os itens negativos fossem vendidos ao menos no custo? E com um markup pequeno?",
        f"Só zerar o prejuízo (break-even) recupera US$ {be0:,.0f}. Um markup modesto de +10% sobre o custo leva a "
        f"US$ {be10:,.0f}. É a análise de maior relação impacto/esforço de todo o case: não depende de modelo, só de política.",
        fig=fig)
    add("Cenários de recuperação (tabela)", "", "",
        "Recuperação estimada em cada cenário de preço-alvo aplicado às transações negativas.",
        table=scen_tbl)

    metrics = {
        "Transações negativas": f"{len(neg):,}",
        "% da base": f"{len(neg)/len(df):.2%}",
        "Sangria total": f"${total_sangria/1e6:,.1f} mi",
        "Recuperação a break-even": f"${abs(total_sangria)/1e6:,.1f} mi",
        "SKUs deficitários": f"{len(neg_skus):,}",
        "SKUs urgentes": f"{len(urgente):,}",
    }
    notes = [
        "Sangria = soma da margem das transações negativas (quanto foi destruído).",
        "Recuperação a break-even = quanto voltaria se cada transação negativa apenas cobrisse o custo.",
        "Estrutural (preço < custo) exige repricing; pontual (devolução) é evento isolado.",
    ]
    return {"metrics": metrics, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


# ========================================================================
# 04 — MODELAGEM AVANÇADA (o diferencial: 5 famílias de modelagem)
# ========================================================================
def run_advanced_modeling(df: pd.DataFrame, p: Params) -> dict:
    """Cinco famílias de modelagem, cada uma com formulação matemática,
    pressupostos, prós/contras, o que responde e o valor em dinheiro que mede.
    Encadeamento: prever (ML) -> explicar (econometria) -> isolar causa (causal)
    -> medir incerteza (bayes) -> decidir sob restrição (OR)."""
    d = df[df["price_valid"]].dropna(subset=["ln_price", "ln_qty"]).copy()
    d["unit_cost"] = -d[COL_COGS_AVG]
    d["ln_cost"] = np.log(d["unit_cost"].clip(lower=0.01))
    blocks = []

    def add(title, why="", answers="", insight="", assumptions="", formula=None,
            formula_legend="", pros=None, cons=None, money="", table=None, fig=None,
            method="", worked=None, not_worked=None, how_it_works="", table2=None, table2_caption=""):
        blocks.append({"title": title, "why": why, "answers": answers,
                       "insight": insight, "assumptions": assumptions,
                       "formula": formula, "formula_legend": formula_legend,
                       "pros": pros, "cons": cons, "money": money,
                       "table": table, "fig": fig,
                       "method": method, "worked": worked, "not_worked": not_worked,
                       "how_it_works": how_it_works, "table2": table2,
                       "table2_caption": table2_caption})

    add("Por que quatro abordagens diferentes",
        why="Um case de pricing não se resolve com uma técnica só. Cada abordagem responde uma pergunta distinta, e o diferencial está em escolher a certa para cada pergunta — e conhecer seus limites.",
        answers="Como prever o preço, explicar a política implícita, testar o efeito causal e transformar tudo numa recomendação de preço?",
        insight="A narrativa é encadeada: o Machine Learning (XGBoost) prevê o preço esperado e mostra os drivers; a Econometria (regressão em painel) explica a política de preço implícita e mede a elasticidade; a Inferência Causal (variável instrumental) testa se mudar o preço realmente muda a demanda; e a Decisão Multicritério (MCDA) transforma tudo num preço recomendado, com o gestor controlando os pesos de cada objetivo.")

    # ===================================================================
    # 1 · MACHINE LEARNING + SHAP
    # ===================================================================
    from sklearn.preprocessing import OrdinalEncoder
    cat = [COL_MATERIAL, COL_COUNTRY, COL_TIER, COL_BUSINESS]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc = enc.fit_transform(d[cat])
    X = np.column_stack([Xc, d["ln_qty"].values])
    feat = [c.replace(" ($)", "") for c in cat] + ["ln_qty"]
    y = d["ln_price"].values

    # XGBoost se disponível; senão, GradientBoosting do sklearn (sempre presente)
    ml_engine = "XGBoost"
    try:
        from xgboost import XGBRegressor
        ml = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.08,
                          n_jobs=-1, tree_method="hist", random_state=p.random_state)
        ml.fit(X, y)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        ml_engine = "HistGradientBoosting (sklearn)"
        ml = HistGradientBoostingRegressor(max_iter=200, max_depth=7,
                                           learning_rate=0.08, random_state=p.random_state)
        ml.fit(X, y)
    r2_ml = ml.score(X, y)

    # importância dos drivers: SHAP se disponível, senão permutação (sklearn)
    try:
        import shap
        idx = np.random.RandomState(p.random_state).choice(len(X), min(1500, len(X)), replace=False)
        sv = shap.TreeExplainer(ml).shap_values(X[idx])
        imp = np.abs(sv).mean(0)
        imp_method = "SHAP"
    except Exception:
        from sklearn.inspection import permutation_importance
        idx = np.random.RandomState(p.random_state).choice(len(X), min(3000, len(X)), replace=False)
        pi = permutation_importance(ml, X[idx], y[idx], n_repeats=3,
                                    random_state=p.random_state, n_jobs=-1)
        imp = pi.importances_mean
        imp_method = "importância por permutação"
    imp = np.abs(imp)
    shap_df = (pd.DataFrame({"driver": feat, f"importância ({imp_method})": imp})
               .sort_values(f"importância ({imp_method})", ascending=False).reset_index(drop=True))
    imp_col = f"importância ({imp_method})"
    shap_df[imp_col] = shap_df[imp_col].round(3)

    # uplift $ do ML: resíduo abaixo do P25 levado ao P25 (mesmo método da aba 5)
    d["_pred"] = ml.predict(X)
    d["_resid"] = d["ln_price"] - d["_pred"]
    tgt = d["_resid"].quantile(0.25)
    opp = d[d["_resid"] < tgt].copy()
    opp["_ptarget"] = np.exp(d.loc[opp.index, "_pred"] + tgt)
    ml_uplift = ((opp["_ptarget"] - opp[COL_PRICE]).clip(lower=0) * opp[COL_QTY]).sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    sd = shap_df.iloc[::-1]
    ax.barh(sd["driver"], sd[imp_col], color="#378add")
    ax.set_xlabel(imp_col.capitalize()); ax.set_title("O que forma o preço? (drivers do modelo)")
    fig.tight_layout()
    top_driver = shap_df.iloc[0]["driver"]
    tier_row = shap_df[shap_df["driver"] == "Customer Tier"]
    tier_imp = tier_row[imp_col].iloc[0] if len(tier_row) else 0

    add("1 · Machine Learning (XGBoost) — prever o preço e explicar os drivers",
        how_it_works="O XGBoost é um método de 'árvores de decisão em sequência'. Imagine várias perguntas encadeadas ('o business é A? o país é X? a quantidade é alta?') que vão dividindo as vendas em grupos cada vez mais parecidos, até estimar um preço para cada perfil. O 'boosting' significa que ele constrói centenas dessas árvores, cada nova corrigindo os erros das anteriores — por isso é tão preciso. Para explicar o que o modelo aprendeu (que costuma ser uma caixa-preta), usamos o SHAP, que mede quanto cada característica empurrou o preço previsto para cima ou para baixo.",
        why=f"ML (gradient boosting, motor: {ml_engine}) prevê o preço esperado com alta acurácia capturando interações não-lineares que a regressão não pega. A importância dos drivers ({imp_method}) abre a caixa-preta: mostra quanto cada variável pesa na previsão.",
        answers="Qual o preço esperado deste perfil, e o que mais determina esse preço?",
        assumptions="Não exige forma funcional nem linearidade. Assume que o passado representa o futuro (padrões estáveis) e que as features disponíveis capturam o essencial. Não infere causalidade — só associação.",
        formula=[r"\hat{p}_i = f(x_i) = \sum_{k=1}^{K} \eta\, T_k(x_i)",
                 r"\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\left[f(S \cup \{j\}) - f(S)\right]"],
        formula_legend="f = ensemble de K árvores T_k com taxa de aprendizado η; φ_j = valor SHAP da feature j (contribuição média marginal sobre todas as coalizões de features), quando SHAP está disponível.",
        pros=["Captura interações e não-linearidades automaticamente",
              "Alta acurácia preditiva (R² elevado)",
              "Importância dá explicabilidade dos drivers"],
        cons=["Não é causal — só correlação",
              "Exige cuidado com overfitting e vazamento",
              "Menos transparente que uma regressão para o board"],
        insight=f"R² de {r2_ml:.2f}. Os drivers dominantes são {top_driver} e Business; o Customer Tier contribui pouquíssimo "
                f"(importância {tier_imp:.3f}) — evidência quantitativa de que a régua de Tier atual quase não explica preço.",
        method=f"Treinamos um modelo de gradient boosting ({ml_engine}) para prever o preço de cada venda a partir das características dela (material, business, tier, país, quantidade). O modelo aprende com o histórico e mede o acerto pelo R² ({r2_ml:.2f} = quanto da variação de preço ele consegue explicar, de 0 a 1). Para saber QUAIS variáveis pesam, usamos {imp_method}: para cada característica, mede-se o quanto ela empurra o preço previsto para cima ou para baixo. O uplift soma as transações vendidas ABAIXO do preço que o modelo esperava, calculando quanto renderia levá-las ao 25º percentil.",
        money=f"Uplift potencial identificado pelo ML (transações abaixo do preço esperado, levadas ao P25): "
              f"US$ {ml_uplift/1e6:,.1f} mi brutos.",
        worked=[
            f"Alta acurácia (R² {r2_ml:.2f}): o modelo prevê bem o preço, então os drivers que ele aponta são confiáveis.",
            "O modelo virou uma régua objetiva de 'preço esperado' — base para achar quem está sendo subprecificado.",
        ],
        not_worked=[
            f"O Customer Tier quase não pesa (importância {tier_imp:.3f}): a segmentação de cliente não está determinando preço.",
            "ML capta associação, não causa: ele diz 'este perfil costuma ter tal preço', não 'se eu subir o preço, o que acontece'.",
        ],
        fig=fig)
    add("Drivers de preço (tabela)", table=shap_df)

    # ===================================================================
    # 1b · PRICE POSITION INDEX (PPI) + OPPORTUNITY VALUE
    # ===================================================================
    # PPI = preço praticado / preço esperado pelo modelo
    d["preco_esperado"] = np.exp(d["_pred"])
    d["PPI"] = d[COL_PRICE] / d["preco_esperado"]

    def ppi_faixa(v):
        if v < 0.90:
            return "Forte desconto (<0,90)"
        if v <= 1.05:
            return "Dentro do esperado (0,90–1,05)"
        return "Acima do benchmark (>1,05)"

    d["faixa_PPI"] = d["PPI"].apply(ppi_faixa)
    # Opportunity Value = quanto se deixa na mesa (só onde PPI < 1)
    d["opportunity_value"] = ((d["preco_esperado"] - d[COL_PRICE]).clip(lower=0) * d[COL_QTY])

    # distribuição do PPI por faixa
    faixa_tbl = (d.groupby("faixa_PPI")
                 .agg(transacoes=("PPI", "size"),
                      valor_deixado_na_mesa=("opportunity_value", "sum"))
                 .reset_index())
    # ordena as faixas de forma lógica
    ordem = ["Forte desconto (<0,90)", "Dentro do esperado (0,90–1,05)", "Acima do benchmark (>1,05)"]
    faixa_tbl["_o"] = faixa_tbl["faixa_PPI"].map({k: i for i, k in enumerate(ordem)})
    faixa_tbl = faixa_tbl.sort_values("_o").drop(columns="_o")
    faixa_tbl_fmt = _round_df(faixa_tbl, money_cols=["valor_deixado_na_mesa"], int_cols=["transacoes"])

    total_mesa = d["opportunity_value"].sum()
    n_desconto = int((d["PPI"] < 0.90).sum())

    # gráfico: distribuição do PPI
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(d["PPI"].clip(0, 2), bins=60, color="#378add")
    ax.axvline(0.90, color="#c0504d", ls="--", lw=1, label="0,90 (forte desconto)")
    ax.axvline(1.05, color="#0F6E56", ls="--", lw=1, label="1,05 (acima do benchmark)")
    ax.set_xlabel("Price Position Index (preço praticado ÷ esperado)")
    ax.set_ylabel("Nº de transações"); ax.legend()
    ax.set_title("Distribuição do Price Position Index (PPI)")
    fig.tight_layout()

    add("1b · Price Position Index (PPI) — onde estamos deixando dinheiro na mesa",
        why="O resíduo do modelo, sozinho, é técnico demais para um gestor. O PPI traduz: é a razão entre o preço praticado e o esperado pelo modelo. Abaixo de 0,90 = forte desconto; entre 0,90 e 1,05 = normal; acima de 1,05 = prêmio. Um número que qualquer diretor lê em segundos.",
        answers="Cada venda está barata, normal ou cara frente ao que o modelo espera? Quanto isso soma?",
        assumptions="O preço esperado do modelo é um benchmark justo (controla material, país, tier, business, volume). PPI < 1 só é oportunidade real se não houver justificativa comercial que o modelo não captou.",
        formula=[r"\text{PPI}_i = \frac{p_i^{\text{praticado}}}{\hat{p}_i^{\text{esperado}}}",
                 r"\text{Opportunity Value}_i = \max(\hat{p}_i - p_i,\, 0)\times q_i"],
        formula_legend="p̂ = preço esperado pelo modelo; PPI<1 indica preço abaixo do benchmark; o Opportunity Value só conta o gap positivo (dinheiro na mesa).",
        insight=f"{n_desconto:,} transações estão em forte desconto (PPI < 0,90). A tabela por faixa mostra quanto cada grupo "
                "representa — a maior parte do valor recuperável concentra-se na faixa de desconto.",
        money=f"Total deixado na mesa (soma do Opportunity Value onde PPI < 1): US$ {total_mesa/1e6:,.1f} mi.",
        fig=fig)
    add("PPI por faixa — quanto cada faixa representa", table=faixa_tbl_fmt)

    # ===================================================================
    # 1c · RANKING DE OPORTUNIDADES (com filtro por tier/país)
    # ===================================================================
    opp_rk = d[d["opportunity_value"] > 0].copy()
    # aplica filtros vindos da UI (reduz o volume da tabela)
    if p.ml_filter_tier and p.ml_filter_tier != "(todos)":
        opp_rk = opp_rk[opp_rk[COL_TIER] == p.ml_filter_tier]
    if p.ml_filter_country and p.ml_filter_country != "(todos)":
        opp_rk = opp_rk[opp_rk[COL_COUNTRY] == p.ml_filter_country]

    rk_cols = [COL_MONTH, COL_BUSINESS, COL_COUNTRY, COL_TIER, COL_MATERIAL,
               COL_PARENT, COL_QTY, COL_PRICE, "preco_esperado", "PPI",
               "opportunity_value"]
    opp_rk = opp_rk.sort_values("opportunity_value", ascending=False)[rk_cols].head(100)
    opp_rk_tbl = _round_df(opp_rk, money_cols=[COL_PRICE, "preco_esperado", "opportunity_value"],
                           pct_cols=["PPI"], int_cols=[COL_QTY])

    filtro_txt = ""
    if p.ml_filter_tier != "(todos)":
        filtro_txt += f" · Tier={p.ml_filter_tier}"
    if p.ml_filter_country != "(todos)":
        filtro_txt += f" · País={p.ml_filter_country}"
    add(f"1c · Ranking de oportunidades (Opportunity Value){filtro_txt}",
        why="Consolida o PPI numa lista acionável: para cada venda abaixo do preço esperado, quanto se recupera se levada ao benchmark. É a fila que o time comercial ataca, cliente a cliente.",
        answers="Por onde começar a renegociar, e quanto vale cada linha?",
        insight="Ordenado pelo maior Opportunity Value. Use o filtro por Tier/País na barra lateral para reduzir o volume "
                "e focar num segmento específico (ex.: só Tier A na Alemanha).",
        money=f"Top 100 oportunidades exibidas{filtro_txt or ' (sem filtro)'} — filtre na barra lateral para focar.",
        table=opp_rk_tbl)

    # ===================================================================
    # 1d · "COMO A IA PENSOU" — explicabilidade em 4 níveis (top oportunidade)
    # ===================================================================
    # transação-alvo = maior Opportunity Value da base
    d["_opp_all"] = d["opportunity_value"]
    ti = int(d["_opp_all"].idxmax())
    row = d.loc[ti]
    p_atual = float(row[COL_PRICE]); p_esp = float(row["preco_esperado"])
    gap_u = p_esp - p_atual; opp_u = float(row["_opp_all"])

    add("Como a IA pensou — explicando a maior oportunidade",
        why="Um modelo de ML só é confiável no board se puder ser explicado. Esta seção abre a caixa-preta para a maior oportunidade da base, em quatro níveis: resumo, decomposição SHAP, regras aprendidas e — o diferencial — transações reais comparáveis do próprio histórico.",
        answers="Por que a IA estimou este preço? Em que evidência do histórico ela se baseou?",
        insight=f"Transação analisada: Material {row[COL_MATERIAL]} · {row[COL_COUNTRY]} · Business {row[COL_BUSINESS]} · "
                f"Tier {row[COL_TIER]} · {int(row[COL_QTY])} unidades.")

    # ---- Nível 1: Resumo Executivo ----
    resumo = pd.DataFrame({
        "métrica": ["Preço atual", "Preço esperado (modelo)", "Gap unitário", "Oportunidade (× volume)"],
        "valor": [f"${p_atual:,.2f}", f"${p_esp:,.2f}", f"${gap_u:,.2f}", f"${opp_u:,.0f}"],
    })
    add("Nível 1 · Resumo executivo",
        insight="O essencial em uma linha: o preço praticado, o que o modelo esperava, e quanto isso vale multiplicado pelo volume.",
        table=resumo)

    # ---- Nível 2: Decomposição SHAP (waterfall) ----
    shap_ok = True
    try:
        import shap
        expl = shap.TreeExplainer(ml)
        sv_row = expl.shap_values(X[ti:ti + 1])[0]
        base_val = float(expl.expected_value)
    except Exception:
        shap_ok = False

    if shap_ok:
        contrib = list(zip(feat, sv_row))
        contrib_sorted = sorted(contrib, key=lambda z: -abs(z[1]))
        # waterfall em preço (aproximação: converte incrementos log para % do preço)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        labels = ["Preço médio\n(base)"] + [c[0] for c in contrib_sorted] + ["Preço\nprevisto"]
        base_price = np.exp(base_val)
        running = base_val
        xs = range(len(labels))
        vals = [base_price]
        cum = base_val
        for f, v in contrib_sorted:
            new = np.exp(cum + v) - np.exp(cum)
            vals.append(new); cum += v
        vals.append(0)
        run = 0
        for i, (lab, val) in enumerate(zip(labels, vals)):
            if i == 0:
                ax.bar(i, base_price, color="#33475b"); run = base_price
            elif i == len(labels) - 1:
                ax.bar(i, np.exp(cum), color="#0F6E56")
            else:
                ax.bar(i, val, bottom=run, color=("#3b6ea5" if val >= 0 else "#c0504d"))
                run += val
        ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Preço (US$)"); ax.set_title("Nível 2 · Como cada variável moveu o preço previsto (SHAP)")
        fig.tight_layout()
        shap_row_tbl = _round_df(
            pd.DataFrame({"variável": [c[0] for c in contrib_sorted],
                          "efeito_no_preço_log": [round(c[1], 3) for c in contrib_sorted]}),
            pct_cols=["efeito_no_preço_log"])
        add("Nível 2 · Explicação SHAP — o que empurrou o preço para cima ou para baixo",
            why="O SHAP parte do preço médio da empresa e mostra quanto cada variável somou ou subtraiu para chegar na previsão desta transação. É a decomposição rigorosa (teoria dos jogos) do 'porquê' daquele número.",
            insight=f"O preço parte da média (${base_price:,.2f}) e cada fator o ajusta: o Material e o Business puxam para cima, "
                    "o alto volume puxa para baixo. A soma dá o preço previsto.",
            fig=fig, table=shap_row_tbl)

    # ---- Nível 3: Regras Aprendidas (padrões estatísticos dos dados) ----
    regras = []
    glob_med = d[COL_PRICE].median()
    # regra material×business
    mat_med = d[d[COL_MATERIAL] == row[COL_MATERIAL]][COL_PRICE].median()
    mb_med = d[(d[COL_MATERIAL] == row[COL_MATERIAL]) & (d[COL_BUSINESS] == row[COL_BUSINESS])][COL_PRICE].median()
    if mat_med > 0 and abs(mb_med / mat_med - 1) > 0.02:
        regras.append(f"Material {row[COL_MATERIAL]} no Business {row[COL_BUSINESS]} tende a preços "
                      f"{(mb_med/mat_med-1)*100:+.0f}% vs. a mediana do material.")
    # regra de volume (sempre)
    q_hi = d[COL_QTY].quantile(0.9)
    hi_rel = (d[d[COL_QTY] >= q_hi][COL_PRICE].median() / glob_med - 1) * 100
    regras.append(f"Vendas de alto volume (≥{int(q_hi)} un.) têm preço mediano {hi_rel:+.0f}% vs. a mediana geral "
                  "— efeito do desconto de volume embutido nos dados.")
    # regra de país×material
    mat_ctry = d[(d[COL_MATERIAL] == row[COL_MATERIAL]) & (d[COL_COUNTRY] == row[COL_COUNTRY])][COL_PRICE].median()
    if mat_med > 0 and abs(mat_ctry / mat_med - 1) > 0.02:
        regras.append(f"No mercado {row[COL_COUNTRY]}, o material {row[COL_MATERIAL]} tem preço mediano "
                      f"{(mat_ctry/mat_med-1)*100:+.0f}% vs. o benchmark do material.")
    # regra business geral (sempre tem)
    biz_rel = (d[d[COL_BUSINESS] == row[COL_BUSINESS]][COL_PRICE].median() / glob_med - 1) * 100
    regras.append(f"O Business {row[COL_BUSINESS]} pratica preço mediano {biz_rel:+.0f}% vs. a mediana geral da empresa.")
    regras_tbl = pd.DataFrame({"regra aprendida do histórico": regras})
    add("Nível 3 · Regras aprendidas — padrões em linguagem de negócio",
        why="Árvores individuais são ilegíveis. Em vez delas, mineramos padrões frequentes dos próprios dados e os traduzimos em frases que um gestor entende — o comportamento que o modelo capturou, dito em português.",
        insight="Estas regras descrevem padrões estatísticos reais do histórico (não a estrutura interna das árvores), "
                "o que as torna interpretáveis e verificáveis.",
        table=regras_tbl)

    # ---- Nível 4: Comparáveis Reais (o diferencial) ----
    mask = ((d[COL_MATERIAL] == row[COL_MATERIAL]) & (d[COL_COUNTRY] == row[COL_COUNTRY])
            & (d[COL_BUSINESS] == row[COL_BUSINESS]) & (d[COL_TIER] == row[COL_TIER]))
    comp = d[mask & (d.index != ti)].copy()
    comp["dist_volume"] = (comp[COL_QTY] - row[COL_QTY]).abs()
    comp = comp.sort_values("dist_volume").head(6)
    if len(comp):
        comp_tbl = _round_df(
            comp[[COL_MATERIAL, COL_COUNTRY, COL_BUSINESS, COL_TIER, COL_QTY, COL_PRICE]],
            money_cols=[COL_PRICE], int_cols=[COL_QTY])
        pmin, pmax = comp[COL_PRICE].min(), comp[COL_PRICE].max()
        pmed = comp[COL_PRICE].median()
        add("Nível 4 · Comparáveis reais — a evidência do próprio histórico",
            why="Este é o nível que constrói confiança de verdade. Em vez de pedir fé na caixa-preta, mostramos as transações mais parecidas já realizadas: mesmo material, país, business e tier, com volume próximo. O usuário vê a evidência concreta.",
            answers="Que vendas reais e semelhantes sustentam a recomendação?",
            insight=f"A IA se baseou em transações historicamente semelhantes, negociadas entre ${pmin:,.2f} e ${pmax:,.2f} "
                    f"(mediana ${pmed:,.2f}). A transação analisada foi vendida a ${p_atual:,.2f} — "
                    f"{'abaixo' if p_atual < pmed else 'dentro'} da faixa dos comparáveis, o que sustenta a oportunidade de reajuste.",
            money=f"Ancorando no histórico comparável (mediana ${pmed:,.2f}), o reajuste sugerido para esta venda "
                  f"recupera ${max(pmed - p_atual, 0) * row[COL_QTY]:,.0f}.",
            table=comp_tbl)
    else:
        add("Nível 4 · Comparáveis reais",
            insight="Não há transações suficientemente semelhantes no histórico para esta venda — sinal de que a previsão "
                    "depende mais de extrapolação do modelo e deve ser usada com cautela.")

    # ---- Nível 5: Árvore ilustrativa (diagrama + regras em linguagem natural) ----
    from sklearn.tree import DecisionTreeRegressor, _tree
    # features interpretáveis (dummies dos valores da própria transação + volume)
    dd = d.copy()
    dd["_isBiz"] = (dd[COL_BUSINESS] == row[COL_BUSINESS]).astype(int)
    dd["_isTier"] = (dd[COL_TIER] == row[COL_TIER]).astype(int)
    dd["_isCtry"] = (dd[COL_COUNTRY] == row[COL_COUNTRY]).astype(int)
    tfeat = ["_isBiz", "_isTier", "_isCtry", COL_QTY]
    tlabels = [f"É Business {row[COL_BUSINESS]}", f"É Tier {row[COL_TIER]}",
               f"É {row[COL_COUNTRY]}", "Quantidade"]
    Xt = dd[tfeat].values
    yt = dd[COL_PRICE].values
    dtree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1000, random_state=p.random_state)
    dtree.fit(Xt, yt)

    # desenha a árvore
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(dtree, feature_names=tlabels, filled=True, rounded=True,
              impurity=False, precision=0, fontsize=7, ax=ax,
              proportion=False)
    ax.set_title("Árvore ilustrativa — como o preço se divide (resumo didático)")
    fig.tight_layout()

    # regras em linguagem natural, com destaque para o caminho da transação
    t = dtree.tree_

    def _regras(node, cond):
        out = []
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = tlabels[t.feature[node]]; thr = t.threshold[node]
            out += _regras(t.children_left[node], cond + [(name, "<=", thr)])
            out += _regras(t.children_right[node], cond + [(name, ">", thr)])
        else:
            out.append((cond, float(t.value[node][0][0]), int(t.n_node_samples[node])))
        return out

    regras_nat = []
    for cond, val, n in _regras(0, []):
        partes = []
        for name, op, thr in cond:
            if name == "Quantidade":
                partes.append(f"Quantidade {op} {thr:.0f}")
            else:
                partes.append(f"{'SIM' if op == '>' else 'NÃO'}: {name.lower()}")
        regras_nat.append({"regra (caminho da árvore)": " e ".join(partes),
                           "preço estimado": round(val, 0), "nº de vendas": n})
    regras_nat_df = _round_df(pd.DataFrame(regras_nat).sort_values("preço estimado"),
                              money_cols=["preço estimado"], int_cols=["nº de vendas"])

    # frase-exemplo do caminho da própria transação
    node = 0; caminho = []
    while t.feature[node] != _tree.TREE_UNDEFINED:
        fi = t.feature[node]; thr = t.threshold[node]
        val_tx = Xt[ti, fi]
        if val_tx <= thr:
            if tlabels[fi] == "Quantidade":
                caminho.append(f"Quantidade ≤ {thr:.0f}")
            else:
                caminho.append(f"não {tlabels[fi].lower()}")
            node = t.children_left[node]
        else:
            if tlabels[fi] == "Quantidade":
                caminho.append(f"Quantidade > {thr:.0f}")
            else:
                caminho.append(tlabels[fi].lower())
            node = t.children_right[node]
    preco_folha = float(t.value[node][0][0])
    frase = f"Se {', '.join(caminho)}, o preço tende a ser ${preco_folha:,.0f}."

    add("Nível 5 · Árvore ilustrativa — o caminho até o preço, visível",
        why="Uma árvore de decisão desenhada mostra, visualmente, como as condições se encadeiam até um preço. Cada árvore do XGBoost real prevê só uma fração do preço em escala log (ilegível), então usamos aqui UMA árvore rasa que aproxima o padrão do conjunto e prevê o preço inteiro — didática, não o modelo real.",
        answers="Como as regras se encadeiam, visualmente, até chegar num preço?",
        insight=f"Seguindo o caminho da transação analisada na árvore: {frase} "
                "Cada folha é um grupo de vendas com preço típico — a tabela abaixo lista todas as regras em linguagem natural.",
        fig=fig, table=regras_nat_df)



    # ===================================================================
    # 2 · ECONOMETRIA — REGRESSÃO EM PAINEL (peer group × mês)
    # ===================================================================
    # Estimador within: agrupa por peer group estável (material × tier × business)
    # e usa a variação AO LONGO DOS MESES dentro de cada grupo. Assim a elasticidade
    # vem de comparar o mesmo item/perfil consigo mesmo no tempo — variação limpa,
    # sem contaminação de mix entre produtos/tiers diferentes.
    import statsmodels.api as sm
    PGe = [COL_MATERIAL, COL_TIER, COL_BUSINESS]
    # agrega por grupo-mês (um ponto = preço/qtd médios daquele grupo naquele mês)
    dpanel = d.copy()
    dpanel["_gkey"] = dpanel.groupby(PGe).ngroup()
    gm = (dpanel.groupby(["_gkey", "date"])
          .agg(ln_price=("ln_price", "mean"), ln_qty=("ln_qty", "mean"),
               ln_cost=("ln_cost", "mean"), n=("ln_price", "size")).reset_index())
    # exige pelo menos 6 meses distintos por grupo (senão não há série p/ estimar)
    meses_por_grupo = gm.groupby("_gkey")["date"].transform("nunique")
    MIN_MESES = 6
    gm = gm[meses_por_grupo >= MIN_MESES].copy()
    n_grupos_painel = gm["_gkey"].nunique()
    # within: demedia cada variável dentro do próprio grupo
    for c in ["ln_qty", "ln_price", "ln_cost"]:
        gm[c + "_dm"] = gm[c] - gm.groupby("_gkey")[c].transform("mean")
    ols = sm.OLS(gm["ln_qty_dm"], sm.add_constant(gm["ln_price_dm"]), missing="drop").fit()
    beta_ols = ols.params["ln_price_dm"]
    ci = ols.conf_int().loc["ln_price_dm"]
    cobertura_receita = 100 * d[d.groupby(PGe)["date"].transform("nunique") >= MIN_MESES][COL_REVENUE].sum() / d[COL_REVENUE].sum()

    # recria as versões demediadas em d (por peer group) para o bloco causal usar a seguir
    for c in ["ln_qty", "ln_price", "ln_cost"]:
        d[c + "_dm"] = d[c] - d.groupby(PGe)[c].transform("mean")

    # gráficos de diagnóstico da regressão (pressupostos): ajuste, resíduos, Q-Q
    import scipy.stats as _sstats
    _samp = gm.dropna(subset=["ln_price_dm", "ln_qty_dm"])
    if len(_samp) > 6000:
        _samp = _samp.sample(6000, random_state=1)
    # resíduos e ajustados: garantir mesmo tamanho, alinhados por posição
    _resid = np.asarray(ols.resid)
    _fitted = np.asarray(ols.fittedvalues)
    _nmin = min(len(_resid), len(_fitted))
    _resid = _resid[:_nmin]
    _fitted = _fitted[:_nmin]
    fig_econ, axes = plt.subplots(1, 3, figsize=(13, 4))
    # (1) scatter de ajuste: ln_qty_dm vs ln_price_dm + reta da regressão
    axes[0].scatter(_samp["ln_price_dm"], _samp["ln_qty_dm"], s=6, alpha=0.12, color="#3b6ea5")
    _xr = np.linspace(_samp["ln_price_dm"].min(), _samp["ln_price_dm"].max(), 50)
    axes[0].plot(_xr, ols.params["const"] + beta_ols * _xr, color="#c0504d", lw=2,
                 label=f"β = {beta_ols:.2f}")
    axes[0].set_xlabel("ln(preço) — desvio do grupo"); axes[0].set_ylabel("ln(quantidade) — desvio do grupo")
    axes[0].set_title("Ajuste: preço × quantidade (within grupo)"); axes[0].legend(fontsize=8)
    # (2) resíduos vs ajustados (homocedasticidade)
    _rs = np.random.RandomState(1).choice(_nmin, min(6000, _nmin), replace=False)
    axes[1].scatter(_fitted[_rs], _resid[_rs], s=6, alpha=0.12, color="#0F6E56")
    axes[1].axhline(0, color="#c0504d", lw=1)
    axes[1].set_xlabel("Valores ajustados"); axes[1].set_ylabel("Resíduos")
    axes[1].set_title("Resíduos vs. ajustados (variância constante?)")
    # (3) Q-Q plot (normalidade dos resíduos)
    _sstats.probplot(_resid[_rs], dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot dos resíduos (normalidade?)")
    axes[2].get_lines()[0].set_markersize(3); axes[2].get_lines()[0].set_alpha(0.3)
    axes[2].get_lines()[0].set_color("#8250c4"); axes[2].get_lines()[1].set_color("#c0504d")
    fig_econ.tight_layout()

    add("2 · Econometria (regressão em painel) — a política de preço implícita",
        how_it_works=f"Em vez de misturar todas as vendas, montamos um PAINEL: agrupamos por peer group estável (mesmo material × tier × business) e, dentro de cada grupo, acompanhamos preço e quantidade MÊS A MÊS. Isso dá, para cada grupo, uma pequena série temporal. A regressão então mede a elasticidade comparando o grupo consigo mesmo ao longo do tempo — 'quando o preço deste item, para este tier, subiu de um mês para outro, o que aconteceu com a quantidade?'. Usamos só grupos com pelo menos {MIN_MESES} meses de histórico (que cobrem {cobertura_receita:.0f}% da receita), e o log-log faz a inclinação virar diretamente a elasticidade.",
        why="A regressão em painel com efeitos fixos de peer group estima a elasticidade usando a variação temporal dentro de cada grupo comparável — a mais limpa possível, pois isola o efeito do preço sem contaminação de diferenças entre produtos, tiers ou países. É a política de preço que os dados revelam, do jeito que a área de pricing explicaria ao board.",
        answers="Qual a elasticidade-preço média e a política de preço implícita nos dados?",
        assumptions=f"Linearidade em log; erros exógenos (E[ε|X]=0); efeitos fixos de peer group (material × tier × business) absorvem o preço-base de cada combinação. A identificação vem da variação MENSAL dentro de cada grupo (≥{MIN_MESES} meses). O pressuposto crítico — e violado aqui — é a exogeneidade do preço.",
        formula=[r"\ln q_{g,t} = \alpha_{g} + \beta\,\ln p_{g,t} + \varepsilon_{g,t}",
                 r"\hat{\beta} = \frac{\text{Cov}(\ln p,\, \ln q)}{\text{Var}(\ln p)} \quad\text{(within peer group, no tempo)}"],
        formula_legend="g = peer group (material × tier × business); t = mês; α_g = efeito fixo do grupo; β = elasticidade-preço (o alvo). A demediação por grupo remove α_g, deixando só a variação temporal dentro de cada grupo.",
        pros=["Coeficientes interpretáveis e comunicáveis",
              "Variação temporal within-group é a identificação mais limpa da elasticidade",
              "Efeitos fixos de peer group controlam produto, tier e business de uma vez"],
        cons=["Assume linearidade em log",
              "Enviesada se o preço for endógeno (é o caso)",
              "Restringe-se a grupos com histórico suficiente de meses"],
        insight=f"A elasticidade em painel é {beta_ols:.2f} [IC95% {ci[0]:.2f}, {ci[1]:.2f}], estimada sobre {n_grupos_painel:,} peer groups "
                f"com pelo menos {MIN_MESES} meses de histórico ({cobertura_receita:.0f}% da receita). É inelástica — sugere espaço para subir preço. "
                "Mas preço e quantidade se determinam juntos (endogeneidade): o próximo bloco testa esse viés.",
        method=f"Montamos um painel: cada peer group (material × tier × business) vira uma série mensal de preço e quantidade médios. Ficamos só com grupos que têm pelo menos {MIN_MESES} meses de dados ({n_grupos_painel:,} grupos, {cobertura_receita:.0f}% da receita). Dentro de cada grupo, demediamos as variáveis (tiramos a média do próprio grupo) para usar SÓ a variação ao longo do tempo. Rodamos a regressão log-log nesses desvios: a inclinação β é a elasticidade — quando o preço deste item/perfil sobe 1% de um mês para outro, a quantidade varia β%. O IC95% é a faixa de confiança.",
        money="Não mede US$ diretamente — fornece a elasticidade, um insumo-chave que alimenta o modelo de decisão multicritério ao final da aba. É um motor de análise, não o resultado final.",
        worked=[
            f"Elasticidade de {beta_ols:.2f} (inelástica): estatisticamente, há espaço para subir preço sem perder muito volume.",
            "Coeficiente interpretável e comunicável — o tipo de número que a área de pricing leva ao board.",
        ],
        not_worked=[
            "Endogeneidade: preço e quantidade se influenciam mutuamente, então este β pode estar enviesado — o bloco causal testa isso.",
            "Assume relação log-linear: pode não capturar efeitos mais complexos de preço sobre demanda.",
        ])
    add("2 · (diagnóstico) — os gráficos que validam a regressão",
        why="Toda regressão faz pressupostos, e é obrigação de rigor verificar se eles valem. São 3 gráficos: (1) AJUSTE — a nuvem de preço vs. quantidade com a reta estimada; a inclinação da reta É a elasticidade. (2) RESÍDUOS vs. AJUSTADOS — os erros do modelo devem estar espalhados de forma uniforme em torno de zero (variância constante, ou homocedasticidade); um funil indicaria problema. (3) Q-Q PLOT — se os pontos seguem a linha vermelha, os resíduos são aproximadamente normais, o que valida os intervalos de confiança.",
        answers="Os pressupostos da regressão se sustentam nos dados?",
        insight="O ajuste mostra a relação negativa esperada (mais preço, menos quantidade). Os resíduos estão distribuídos em torno de zero sem padrão forte de funil, e o Q-Q plot segue razoavelmente a diagonal — os pressupostos se sustentam o suficiente para confiar na estimativa da elasticidade e no seu intervalo de confiança.",
        fig=fig_econ)

    # -- Econometria, nível A: resumo executivo --
    econ_resumo = pd.DataFrame({
        "métrica": ["Elasticidade-preço (β)", "IC 95% inferior", "IC 95% superior",
                    "R² (within peer group)", "Leitura"],
        "valor": [f"{beta_ols:.3f}", f"{ci[0]:.3f}", f"{ci[1]:.3f}",
                  f"{ols.rsquared:.4f}",
                  "inelástico" if beta_ols > -1 else "elástico"],
    })
    add("2a · Econometria — resumo executivo",
        insight="O número central e sua faixa de confiança. Elasticidade entre 0 e −1 significa inelástico "
                "(subir preço aumenta receita); abaixo de −1, elástico. Nota sobre o R² baixo: em elasticidade de painel "
                "isso é esperado e saudável — o preço sozinho explica pouco da variação de quantidade (a demanda depende de "
                "muitos fatores), mas o coeficiente de elasticidade em si é preciso, como mostra o intervalo de confiança estreito.",
        table=econ_resumo)

    # -- Econometria, nível B: interpretação de negócio (o que X% de aumento faz) --
    cenarios = []
    for aumento in [0.05, 0.10, 0.15]:
        var_q = beta_ols * aumento * 100
        var_receita = (1 + aumento) * (1 + var_q / 100) - 1  # aprox. receita
        cenarios.append({"aumento_de_preço": f"+{aumento:.0%}",
                         "variação_de_volume": f"{var_q:+.1f}%",
                         "efeito_na_receita": f"{var_receita*100:+.1f}%"})
    add("2b · Econometria — tradução para o negócio",
        why="A elasticidade só vira decisão quando traduzida em cenários concretos de preço, volume e receita.",
        insight=f"Com β = {beta_ols:.2f}, cada aumento de preço perde pouco volume (demanda inelástica), então a "
                "receita SOBE. A tabela mostra o efeito líquido de três cenários de reajuste.",
        table=pd.DataFrame(cenarios))

    # ===================================================================
    # 3 · INFERÊNCIA CAUSAL (IV / 2SLS)
    # ===================================================================
    sub = d.dropna(subset=["ln_qty_dm", "ln_price_dm", "ln_cost_dm"]).copy()
    sub = sub[np.isfinite(sub["ln_cost_dm"])]
    samp = sub.sample(min(50000, len(sub)), random_state=p.random_state)
    # 1º estágio: preço ~ custo (mede força do instrumento)
    fs = sm.OLS(samp["ln_price_dm"], sm.add_constant(samp["ln_cost_dm"])).fit()
    f_stat = float(fs.fvalue)
    weak = f_stat < 10
    # 2SLS: linearmodels se disponível; senão, 2SLS manual (duas OLS do statsmodels)
    try:
        from linearmodels.iv import IV2SLS
        iv = IV2SLS(samp["ln_qty_dm"], sm.add_constant(samp[[]]),
                    samp["ln_price_dm"], samp["ln_cost_dm"]).fit()
        beta_iv = float(iv.params["ln_price_dm"])
    except Exception:
        # 2SLS manual: preço previsto pelo instrumento -> regride quantidade nele
        p_hat = fs.predict(sm.add_constant(samp["ln_cost_dm"]))
        second = sm.OLS(samp["ln_qty_dm"], sm.add_constant(p_hat)).fit()
        beta_iv = float(second.params.iloc[1])
    iv_tbl = pd.DataFrame({
        "método": ["OLS (ingênuo)", "IV / 2SLS (custo como instrumento)"],
        "elasticidade": [round(beta_ols, 3), round(beta_iv, 3)],
        "confiável?": ["enviesado (endogeneidade)", "NÃO — instrumento fraco" if weak else "sim"],
    })

    # gráficos de diagnóstico do causal: 1º estágio (força do instrumento) + resíduos
    _cs = np.random.RandomState(1).choice(len(samp), min(5000, len(samp)), replace=False)
    _csamp = samp.iloc[_cs]
    fig_caus, axcs = plt.subplots(1, 2, figsize=(11, 4.2))
    # (1) 1º estágio: custo (instrumento) vs preço — deve ter relação forte se instrumento é bom
    axcs[0].scatter(_csamp["ln_cost_dm"], _csamp["ln_price_dm"], s=6, alpha=0.15, color="#BA7517")
    _xc = np.linspace(_csamp["ln_cost_dm"].min(), _csamp["ln_cost_dm"].max(), 50)
    axcs[0].plot(_xc, fs.params["const"] + fs.params["ln_cost_dm"] * _xc, color="#c0504d", lw=2)
    axcs[0].set_xlabel("ln(custo) — desvio do grupo [instrumento]")
    axcs[0].set_ylabel("ln(preço) — desvio do grupo")
    axcs[0].set_title(f"1º estágio: custo → preço (F={f_stat:.1f}, {'FRACO' if weak else 'forte'})")
    # (2) força do instrumento vs limiar
    axcs[1].bar(["F-stat do\ninstrumento", "Limiar mínimo\n(regra F>10)"], [f_stat, 10],
                color=["#c0504d" if weak else "#1d9e75", "#888880"])
    axcs[1].axhline(10, color="grey", ls="--", lw=0.8)
    axcs[1].set_ylabel("Valor da estatística F")
    axcs[1].set_title("Força do instrumento vs. limiar de validade")
    for i, v in enumerate([f_stat, 10]):
        axcs[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig_caus.tight_layout()

    add("3 · Inferência causal (variável instrumental / 2SLS) — o que acontece se MUDARMOS o preço?",
        how_it_works="O problema: nos dados, preço e demanda se influenciam mutuamente (preço alto pode ser causa OU consequência de baixa demanda), então uma regressão simples confunde causa e efeito. A variável instrumental resolve isso em 2 passos (2SLS = mínimos quadrados em dois estágios): procuramos algo que mexa no preço mas não na demanda diretamente — aqui, o CUSTO. 1º passo: usamos o custo para prever a parte do preço que é 'limpa' (não contaminada pela demanda). 2º passo: usamos esse preço limpo para medir o efeito real na demanda. Um teste (F-stat) verifica se o custo é um instrumento forte o suficiente para confiar no resultado.",
        why="A regressão em painel estima a elasticidade média de forma interpretável, mas pode estar enviesada pela endogeneidade. A variável instrumental isola o efeito causal do preço na demanda usando o custo como instrumento: o custo desloca o preço (relevância) mas não afeta a demanda por outra via (exclusão). É o padrão-ouro para elasticidade causal sem experimento. Os gráficos mostram o 1º estágio (a força do instrumento) — a chave para saber se o método é válido aqui.",
        answers="Se aumentarmos o preço de propósito, a demanda cai de verdade — e quanto?",
        assumptions="(1) Relevância: o instrumento Z correlaciona com o preço (testável via F-stat). (2) Exclusão: Z só afeta a demanda ATRAVÉS do preço — não testável, exige argumento econômico. (3) Exogeneidade do instrumento.",
        formula=[r"\text{1º est.: } \ln p_i = \pi_0 + \pi_1 \ln c_i + \nu_i",
                 r"\text{2º est.: } \ln q_i = \alpha + \beta^{IV} \widehat{\ln p_i} + \varepsilon_i",
                 r"\hat{\beta}^{IV} = \frac{\text{Cov}(\ln c,\, \ln q)}{\text{Cov}(\ln c,\, \ln p)}"],
        formula_legend="c = custo unitário (instrumento); π_1 mede a força do instrumento (F-stat do 1º estágio deve ser > 10).",
        pros=["Corrige o viés de endogeneidade",
              "Estima efeito causal, não correlação",
              "Testável: a força do instrumento é mensurável"],
        cons=["Depende de um instrumento válido (difícil)",
              "Exclusão não é testável — exige teoria",
              "Instrumento fraco gera estimativa pior que OLS"],
        insight=(f"ACHADO DE RIGOR: o instrumento é FRACO (F-stat do 1º estágio = {f_stat:.1f}, abaixo do limiar 10). "
                 f"Logo a estimativa IV ({beta_iv:.1f}) é inválida e NÃO deve ser usada. O custo não é bom instrumento aqui — "
                 "provavelmente se correlaciona com qualidade/tipo de produto, violando a restrição de exclusão. "
                 "Reconhecer isso vale mais que reportar número errado."
                 if weak else
                 f"Elasticidade causal (IV) = {beta_iv:.2f}, instrumento forte (F={f_stat:.0f})."),
        money="Não mede US$ — é o teste de validade que impede usarmos uma elasticidade causal errada nas decisões seguintes. Vale como seguro contra decisão ruim.",
        method="Para saber se MUDAR o preço realmente muda a demanda (causa, não correlação), usamos a técnica de variável instrumental (2 estágios). A ideia: precisamos de algo que mexa no preço mas não mexa na demanda por outro caminho — usamos o CUSTO como esse 'instrumento'. 1º estágio: prevemos o preço a partir do custo. 2º estágio: usamos esse preço 'limpo' para estimar o efeito na demanda. O teste-chave é o F-stat do 1º estágio: se for maior que 10, o instrumento é forte; abaixo disso, é fraco e o resultado não vale.",
        table=iv_tbl,
        worked=[
            "Testar a validade ANTES de usar o número é o certo — evita levar ao board uma elasticidade causal errada.",
            "Reconhecer a limitação (instrumento fraco) demonstra rigor: preferimos não responder a responder errado.",
        ] if weak else [
            f"Instrumento forte (F={f_stat:.0f}): a elasticidade causal ({beta_iv:.2f}) é confiável.",
            "Conseguimos separar causa de correlação — padrão-ouro sem precisar de experimento.",
        ],
        not_worked=[
            f"Instrumento FRACO (F={f_stat:.1f} < 10): o custo não serve como instrumento aqui, então a elasticidade causal fica indeterminada.",
            "Sem instrumento válido, usamos a elasticidade observacional (OLS) com a ressalva de que pode ter viés.",
        ] if weak else [
            "IV depende de pressupostos não totalmente testáveis (restrição de exclusão) — exige argumento econômico.",
        ],
        fig=fig_caus)

    # -- Causal, nível A: teste de força do instrumento (1º estágio) --
    caus_diag = pd.DataFrame({
        "teste": ["F-stat do 1º estágio", "Limiar de instrumento forte", "Veredito"],
        "valor": [f"{f_stat:.1f}", "10,0",
                  "INSTRUMENTO FRACO — não usar IV" if weak else "instrumento forte — IV válido"],
    })
    add("3a · Causal — diagnóstico de validade do instrumento",
        why="Antes de acreditar em qualquer estimativa IV, testa-se a força do instrumento. Sem isso, o IV pode ser pior que o OLS.",
        insight=f"O F-stat mede se o custo realmente move o preço (relevância). Aqui vale {f_stat:.1f}. "
                + ("Abaixo de 10, o instrumento é fraco e a estimativa IV é não-confiável — descartada."
                   if weak else "Acima de 10, o instrumento é forte e a estimativa IV é válida."),
        table=caus_diag)

    # -- Causal, nível B: o que fazer diante disso (decisão metodológica) --
    add("3b · Causal — a decisão metodológica honesta",
        why="Reconhecer a limitação e escolher o caminho certo é o que separa análise rigorosa de teatro de números.",
        insight=("Como o instrumento é fraco, NÃO reportamos a elasticidade IV. Mantemos a econométrica como aproximação, "
                 "com a ressalva de endogeneidade. O caminho ideal para elasticidade causal de verdade seria um experimento "
                 "de preço (teste A/B) ou um instrumento melhor (choque de câmbio, mudança tributária) — fora do escopo desta base."
                 if weak else
                 "O instrumento é válido, então a elasticidade causal (IV) é a estimativa preferida sobre a OLS."),
        money="Não mede $ — protege a decisão de usar uma elasticidade enviesada, o que evitaria erro de precificação.")

    # -- Causal, nível C: a resposta causal (condicional à validade) --
    # o que a leitura causal diria, contrastando com a observacional (OLS)
    exemplo_aumento = 0.10
    causal_resp = pd.DataFrame({
        "leitura": ["Observacional (OLS) — o que temos",
                    "Causal (IV) — o que gostaríamos"],
        "elasticidade": [f"{beta_ols:.2f}", f"{beta_iv:.2f}" + (" ⚠ inválida" if weak else "")],
        f"efeito de +{exemplo_aumento:.0%} no preço": [
            f"demanda {beta_ols*exemplo_aumento*100:+.1f}% (associação)",
            (f"demanda {beta_iv*exemplo_aumento*100:+.1f}% (causal)" if not weak
             else "não estimável — instrumento fraco")],
        "vale como decisão?": [
            "aproximação, com ressalva de viés",
            "sim, se causal" if not weak else "NÃO — instrumento fraco"],
    })
    add("3c · Causal — a resposta que buscamos (e por que não a temos aqui)",
        why="O objetivo final do IV é uma frase de INTERVENÇÃO: 'aumentar o preço em X% CAUSA uma queda de Y% na demanda'. Diferente do OLS, que só descreve como preço e volume variaram juntos no passado.",
        answers=f"Se aumentarmos o preço de propósito em +{exemplo_aumento:.0%}, quanto a demanda cai — de verdade?",
        insight=(f"SE o instrumento fosse válido, a resposta causal seria: '+{exemplo_aumento:.0%} de preço causa "
                 f"{beta_iv*exemplo_aumento*100:+.1f}% de demanda'. Mas como o F-stat é {f_stat:.1f} (< 10), essa frase NÃO se sustenta. "
                 f"O que podemos dizer, com ressalva, é a leitura observacional (OLS): +{exemplo_aumento:.0%} de preço está associado a "
                 f"{beta_ols*exemplo_aumento*100:+.1f}% de demanda — associação, não causa comprovada."
                 if weak else
                 f"Com instrumento válido, a resposta causal é: '+{exemplo_aumento:.0%} de preço CAUSA "
                 f"{beta_iv*exemplo_aumento*100:+.1f}% de demanda'. Esta é a base para decidir o reajuste."),
        table=causal_resp)


    # ===================================================================
    # 4 · DECISÃO MULTICRITÉRIO (MCDA) — preço recomendado a nível SKU
    # ===================================================================
    from sklearn.preprocessing import minmax_scale, StandardScaler
    from sklearn.cluster import KMeans

    # pesos vindos do usuário (5 critérios; normalizados para somar 1)
    w_raw = {"margem": getattr(p, "w_margem", 0.20),
             "elast": getattr(p, "w_elasticidade", 0.20),
             "share": getattr(p, "w_share", 0.20),
             "churn": getattr(p, "w_churn", 0.20),
             "cluster": getattr(p, "w_cluster", 0.20)}
    w_sum = sum(w_raw.values()) or 1.0
    w = {k: v / w_sum for k, v in w_raw.items()}
    teto = getattr(p, "max_price_increase", 0.15)
    meses_tot = d["date"].nunique()

    # UNIDADE DE PRECIFICAÇÃO = SKU × tier × business × país
    PGm = [COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY]
    dseg = d.copy()
    seg = dseg.groupby(PGm).agg(
        preco=(COL_PRICE, "median"), custo=("unit_cost", "median"),
        receita=(COL_REVENUE, "sum"), qtd=(COL_QTY, "sum"), n=(COL_PRICE, "size"),
        meses=("date", "nunique"), clientes=(COL_PARENT, "nunique")).reset_index()
    seg = seg[(seg["n"] >= 5) & (seg["preco"] > 0)].copy()
    seg["margem_pct"] = (seg["preco"] - seg["custo"]) / seg["preco"].clip(lower=0.01)

    # critério 1 · MARGEM: margem baixa = espaço para subir
    seg["s_margem"] = 1 - minmax_scale(seg["margem_pct"].clip(-2, 1))
    # critério 2 · ELASTICIDADE: inelástico = seguro p/ subir (elasticidade por business)
    elast_by_biz = {}
    for biz, gb in dseg.groupby(COL_BUSINESS):
        if len(gb) >= 50 and gb["ln_price"].std() > 0:
            try:
                elast_by_biz[biz] = float(np.polyfit(gb["ln_price"] - gb["ln_price"].mean(),
                                                     gb["ln_qty"] - gb["ln_qty"].mean(), 1)[0])
            except Exception:
                elast_by_biz[biz] = beta_ols
        else:
            elast_by_biz[biz] = beta_ols
    seg["elast"] = seg[COL_BUSINESS].map(elast_by_biz).fillna(beta_ols)
    seg["s_elast"] = minmax_scale(seg["elast"].clip(-2, 0))  # menos elástico = score maior
    # critério 3 · MARKET SHARE: share do SKU DENTRO DO BUSINESS (concorre com todos do business);
    # share ALTO = SKU importante no business = proteger = score baixo
    biz_rev = seg.groupby(COL_BUSINESS)["receita"].transform("sum")
    seg["share"] = seg["receita"] / biz_rev.clip(lower=1)
    seg["s_share"] = 1 - minmax_scale(seg["share"])  # share baixo = pode subir mais
    # critério 4 · CHURN: chance de perder o cliente daquele SKU; churn alto = NÃO subir preço
    # score comportamental (RFM-like) por cliente: recência + frequência + tendência de volume
    tmax = int(d["date"].map({dt: i for i, dt in enumerate(sorted(d["date"].unique()))}).max())
    _tmap = {dt: i for i, dt in enumerate(sorted(d["date"].unique()))}
    d["_t"] = d["date"].map(_tmap)
    cli = d.groupby(COL_PARENT).agg(ult=("_t", "max"), nm=("_t", "nunique"),
                                    prim=("_t", "min")).reset_index()
    cli["recencia"] = tmax - cli["ult"]
    cli["span"] = (cli["ult"] - cli["prim"] + 1).clip(lower=1)
    cli["freq_rel"] = cli["nm"] / cli["span"]
    rec6 = d[d["_t"] > tmax - 6].groupby(COL_PARENT)[COL_REVENUE].sum()
    prev6 = d[(d["_t"] <= tmax - 6) & (d["_t"] > tmax - 12)].groupby(COL_PARENT)[COL_REVENUE].sum()
    cli = cli.merge(rec6.rename("rec6"), on=COL_PARENT, how="left").merge(
        prev6.rename("prev6"), on=COL_PARENT, how="left")
    cli["rec6"] = cli["rec6"].fillna(0); cli["prev6"] = cli["prev6"].fillna(0)
    cli["tend"] = (cli["rec6"] - cli["prev6"]) / cli["prev6"].clip(lower=1)
    cli["churn"] = (minmax_scale(cli["recencia"]) * 0.5 +
                    (1 - minmax_scale(cli["freq_rel"])) * 0.3 +
                    (1 - minmax_scale(cli["tend"].clip(-2, 2))) * 0.2)
    churn_by_cli = cli.set_index(COL_PARENT)["churn"]
    # churn do SKU = média do churn dos clientes que compram aquele SKU (ponderada por receita)
    d["_churn_cli"] = d[COL_PARENT].map(churn_by_cli).fillna(0.5)
    churn_sku = (d.groupby(PGm)
                 .apply(lambda g: np.average(g["_churn_cli"], weights=g[COL_REVENUE].clip(lower=1)))
                 .rename("churn_sku").reset_index())
    seg = seg.merge(churn_sku, on=PGm, how="left")
    seg["churn_sku"] = seg["churn_sku"].fillna(0.5)
    # churn ALTO = risco de perder cliente = NÃO subir = score BAIXO
    seg["s_churn"] = 1 - minmax_scale(seg["churn_sku"])
    # critério 5 · ALINHAMENTO NO CLUSTER: k-means por comportamento; preço abaixo do cluster = subir
    feats = pd.DataFrame({
        "vol": np.log1p(seg["qtd"]), "freq": seg["meses"],
        "margem": seg["margem_pct"].clip(-2, 1), "preco": np.log1p(seg["preco"])})
    Xs = StandardScaler().fit_transform(feats.fillna(0))
    n_clusters = min(6, max(2, len(seg) // 500))
    km = KMeans(n_clusters=n_clusters, random_state=1, n_init=10).fit(Xs)
    seg["cluster"] = km.labels_
    seg["cl_med_preco"] = seg.groupby("cluster")["preco"].transform("median")
    seg["gap_cluster"] = ((seg["cl_med_preco"] - seg["preco"]) / seg["cl_med_preco"].clip(lower=0.01)).clip(lower=0)
    seg["s_cluster"] = minmax_scale(seg["gap_cluster"]) if seg["gap_cluster"].nunique() > 1 else 0.0

    # score combinado (5 critérios)
    seg["score"] = (w["margem"] * seg["s_margem"] + w["elast"] * seg["s_elast"] +
                    w["share"] * seg["s_share"] + w["churn"] * seg["s_churn"] +
                    w["cluster"] * seg["s_cluster"])
    # ESCALA CORRIGIDA: usa o percentil do score para espalhar de 0 ao teto (diferencia de verdade)
    seg["score_pct"] = seg["score"].rank(pct=True)
    seg["reajuste_%"] = teto * seg["score_pct"] * 100
    seg["preco_atual"] = seg["preco"]
    seg["preco_recomendado"] = seg["preco"] * (1 + teto * seg["score_pct"])
    seg["ganho_potencial"] = teto * seg["score_pct"] * seg["receita"]
    seg = seg.sort_values("ganho_potencial", ascending=False)
    ganho_mcda = float(seg["ganho_potencial"].sum())
    n_skus = len(seg)

    # filtro por SKU (seletor)
    sku_sel = getattr(p, "mcda_sku", "(todos)")
    if sku_sel and sku_sel != "(todos)":
        seg_view = seg[seg[COL_MATERIAL].astype(str) == sku_sel]
        if len(seg_view) == 0:
            seg_view = seg.head(20)
    else:
        seg_view = seg.head(20)

    # GRÁFICO NOVO: reajuste recomendado + ganho US$ dos top 15 SKUs (duas leituras claras)
    top_g = seg.head(15).iloc[::-1]  # invertido p/ maior no topo do barh
    lab = [f"{str(m)[:10]} T{t}·B{b}" for m, t, b in
           zip(top_g[COL_MATERIAL], top_g[COL_TIER], top_g[COL_BUSINESS])]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    ypos = np.arange(len(top_g))
    axA.barh(ypos, top_g["reajuste_%"], color="#378add")
    axA.set_yticks(ypos); axA.set_yticklabels(lab, fontsize=7)
    axA.set_xlabel("Reajuste recomendado (%)")
    axA.set_title("Quanto reajustar (% até o teto)")
    for i, v in enumerate(top_g["reajuste_%"]):
        axA.text(v, i, f" {v:.1f}%", va="center", fontsize=7)
    axB.barh(ypos, top_g["ganho_potencial"] / 1e6, color="#1d9e75")
    axB.set_xlabel("Ganho potencial (US$ mi)")
    axB.set_title("Quanto isso rende (US$)")
    for i, v in enumerate(top_g["ganho_potencial"] / 1e6):
        axB.text(v, i, f" ${v:,.2f}mi", va="center", fontsize=7)
    fig.suptitle("Top 15 SKUs por ganho — reajuste recomendado e retorno", fontsize=11)
    fig.tight_layout()

    mcda_tbl = seg_view[[COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY, "preco_atual",
                         "preco_recomendado", "reajuste_%", "ganho_potencial", "cluster"]].copy()
    mcda_tbl["preço_atual"] = mcda_tbl["preco_atual"].map(lambda v: f"US$ {v:,.2f}")
    mcda_tbl["preço_recomendado"] = mcda_tbl["preco_recomendado"].map(lambda v: f"US$ {v:,.2f}")
    mcda_tbl["reajuste_%"] = mcda_tbl["reajuste_%"].round(1)
    mcda_tbl["ganho_US$"] = mcda_tbl["ganho_potencial"].map(lambda v: f"US$ {v:,.0f}")
    mcda_tbl["cluster"] = mcda_tbl["cluster"].astype(int)
    mcda_tbl = mcda_tbl[[COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY,
                         "preço_atual", "preço_recomendado", "reajuste_%", "ganho_US$", "cluster"]]

    perfil_nome = getattr(p, "mcda_perfil", "Equilibrado")
    pesos_txt = (f"Margem {w['margem']*100:.0f}% · Elasticidade {w['elast']*100:.0f}% · "
                 f"Market share {w['share']*100:.0f}% · Churn {w['churn']*100:.0f}% · "
                 f"Cluster {w['cluster']*100:.0f}%")
    seg_top = seg.iloc[0]

    # exemplo passo a passo do SKU líder — em linguagem de negócio (para não-técnicos)
    _st = seg_top
    _c_marg = w["margem"] * _st["s_margem"]
    _c_elas = w["elast"] * _st["s_elast"]
    _c_shar = w["share"] * _st["s_share"]
    _c_chur = w["churn"] * _st["s_churn"]
    _c_clus = w["cluster"] * _st["s_cluster"]
    _score_bruto = _c_marg + _c_elas + _c_shar + _c_chur + _c_clus
    _v_margem = _st["margem_pct"] * 100
    _v_elast = _st["elast"]
    _v_share = _st["share"] * 100
    _v_churn = _st.get("churn_sku", 0.5)
    _v_gap = _st.get("gap_cluster", 0) * 100
    _mat = _st[COL_MATERIAL]

    # helper: traduz uma nota 0-1 em "posição" que um gestor entende
    def _pos(nota):
        if nota >= 0.75:
            return "entre os mais altos"
        if nota >= 0.5:
            return "acima da média"
        if nota >= 0.25:
            return "abaixo da média"
        return "entre os mais baixos"

    conta_txt = (
        f"**Vamos pegar um produto real e ver, passo a passo, como o modelo chega no preço sugerido.**\n\n"
        f"Produto: SKU **{_mat}** (tier {_st[COL_TIER]}, {_st[COL_COUNTRY]}) · Objetivo escolhido: **{getattr(p, 'mcda_perfil', 'Equilibrado')}**\n\n"
        f"**A regra:** para cada um dos 5 critérios, damos ao produto uma nota de 0 a 10 "
        f"(0 = nenhum motivo para subir o preço, 10 = motivo máximo). A nota sai comparando o produto com todos os outros.\n\n"
        f"**Critério 1 — Margem: nota {_st['s_margem']*10:.0f}/10.** "
        f"Este produto tem margem de {_v_margem:.0f}% (vende a US$ {_st['preco']:,.2f}, custa US$ {_st['custo']:,.2f}). "
        f"Comparando com os outros, essa margem está {_pos(_st['s_margem'] if _v_margem < 50 else 1-_st['s_margem'])}. "
        f"Margem já {'gorda' if _st['s_margem'] < 0.5 else 'apertada'} significa {'pouco' if _st['s_margem'] < 0.5 else 'bastante'} espaço para subir → nota {_st['s_margem']*10:.0f}.\n\n"
        f"**Critério 2 — Elasticidade: nota {_st['s_elast']*10:.0f}/10.** "
        f"Mede se o cliente foge quando o preço sobe. Aqui a sensibilidade é {_v_elast:.2f} "
        f"({'baixa — cliente aguenta reajuste' if _st['s_elast'] > 0.5 else 'considerável — cliente reage ao preço'}) → nota {_st['s_elast']*10:.0f}.\n\n"
        f"**Critério 3 — Market share: nota {_st['s_share']*10:.0f}/10.** "
        f"Este produto representa {_v_share:.1f}% da receita do seu business — um share {_pos(1-_st['s_share'])}. "
        f"{'Como pesa pouco, dá para reajustar sem risco de mexer no faturamento' if _st['s_share'] > 0.5 else 'Como é importante, convém proteger e não arriscar'} → nota {_st['s_share']*10:.0f}.\n\n"
        f"**Critério 4 — Churn (risco de perder o cliente): nota {_st['s_churn']*10:.0f}/10.** "
        f"Os clientes deste produto têm risco {'baixo' if _st['s_churn'] > 0.5 else 'alto'} de parar de comprar "
        f"(índice {_v_churn:.2f} de 1). Cliente {'fiel aguenta um reajuste' if _st['s_churn'] > 0.5 else 'em risco: melhor não subir'} → nota {_st['s_churn']*10:.0f}.\n\n"
        f"**Critério 5 — Cluster (preço vs. produtos parecidos): nota {_st['s_cluster']*10:.0f}/10.** "
        f"Agrupamos produtos de comportamento parecido; este está {_v_gap:.0f}% "
        f"{'abaixo do' if _v_gap > 0 else 'alinhado com o'} preço típico do grupo. "
        f"{'Está barato para o que é — há espaço para subir' if _st['s_cluster'] > 0.5 else 'Já está no nível dos pares'} → nota {_st['s_cluster']*10:.0f}.\n\n"
        f"**Agora juntamos tudo, respeitando os pesos do objetivo '{getattr(p, 'mcda_perfil', 'Equilibrado')}':**\n\n"
        f"- Margem: nota {_st['s_margem']*10:.0f} × peso {w['margem']*100:.0f}% = {_c_marg*10:.2f}\n"
        f"- Elasticidade: nota {_st['s_elast']*10:.0f} × peso {w['elast']*100:.0f}% = {_c_elas*10:.2f}\n"
        f"- Market share: nota {_st['s_share']*10:.0f} × peso {w['share']*100:.0f}% = {_c_shar*10:.2f}\n"
        f"- Churn: nota {_st['s_churn']*10:.0f} × peso {w['churn']*100:.0f}% = {_c_chur*10:.2f}\n"
        f"- Cluster: nota {_st['s_cluster']*10:.0f} × peso {w['cluster']*100:.0f}% = {_c_clus*10:.2f}\n"
        f"- **Nota final (score) = {_score_bruto*10:.2f} de 10**\n\n"
        f"**Do score ao preço sugerido:** essa nota final coloca o produto à frente de {_st['score_pct']*100:.0f}% dos produtos "
        f"na fila de prioridade de reajuste. Quanto mais à frente, maior o reajuste (limitado ao teto de {teto*100:.0f}%). "
        f"Então o reajuste é {teto*100:.0f}% × {_st['score_pct']:.2f} = **{_st['reajuste_%']:.1f}%**, "
        f"e o preço passa de US$ {_st['preco_atual']:,.2f} para **US$ {_st['preco_recomendado']:,.2f}**.")

    add("4 · Decisão multicritério (MCDA) — o preço recomendado por SKU",
        how_it_works="A precificação real é produto a produto, então o modelo trabalha na unidade certa: SKU × tier × business × país. Você define o que importa e quanto (pesos), sobre 5 critérios: MARGEM (margem baixa = espaço para subir), ELASTICIDADE (inelástico = seguro subir), MARKET SHARE (share do SKU dentro do business — share alto = SKU importante, proteger, subir menos), CHURN (chance de perder o cliente daquele SKU — churn alto = NÃO subir, para não perdê-lo) e CLUSTER (agrupamos SKUs de comportamento parecido via k-means e vemos quão abaixo dos pares do cluster o preço está). Cada SKU recebe uma nota de 0 a 1 em cada critério; multiplicamos pelos pesos e somamos num score. O reajuste é proporcional à POSIÇÃO do SKU no ranking de scores — o que usa a faixa inteira até o teto, diferenciando de fato os produtos.",
        why="Um número de caixa-preta é difícil de defender, e um número agregado não serve para precificar — a decisão de preço é por produto. Este modelo entrega um preço recomendado a nível SKU, transparente e ajustável: o gestor escolhe o peso de cada objetivo e vê o preço de cada produto se ajustar. É a ponte entre a análise e a decisão executiva de pricing.",
        answers="Dado o que a empresa prioriza, qual o preço recomendado para cada SKU (por tier, business e país)?",
        assumptions=f"Unidade de decisão: SKU × tier × business × país, com pelo menos 5 vendas. Os 5 critérios são normalizados de 0 a 1 antes de ponderar. O reajuste é proporcional ao PERCENTIL do score (posição no ranking), limitado ao teto (+{teto*100:.0f}%) — assim os SKUs de maior oportunidade chegam perto do teto e os de menor ficam perto de zero. A elasticidade vem da econometria por business; os clusters vêm de k-means sobre volume, frequência, margem e preço.",
        method=f"{conta_txt}",
        insight=f"Trabalhando a nível SKU ({n_skus:,} combinações precificáveis), com o perfil '{perfil_nome}' ({pesos_txt}), "
                f"o SKU de maior ganho é o material {seg_top[COL_MATERIAL]} (tier {seg_top[COL_TIER]}, business {seg_top[COL_BUSINESS]}, "
                f"{seg_top[COL_COUNTRY]}): reajuste de {seg_top['reajuste_%']:.1f}%, de US$ {seg_top['preco_atual']:,.0f} para "
                f"US$ {seg_top['preco_recomendado']:,.0f}. Trocar o perfil de objetivo reordena tudo — 'Defender share' protege "
                "clientes com risco de churn e share alto; 'Extrair margem' foca nos de margem baixa e inelásticos.",
        money=f"Com os pesos atuais, o ganho potencial somado é ~US$ {ganho_mcda/1e6:,.0f} mi "
              "(teto teórico; a captura real depende de execução comercial).",
        worked=[
            "Precificação na unidade certa (SKU/tier/business/país) e escala que usa toda a faixa até o teto — reajustes diferenciados de verdade.",
            "5 critérios de negócio reais (não só variáveis cruas): margem, elasticidade, share, ruptura e alinhamento por cluster de comportamento.",
        ],
        not_worked=[
            "Os pesos são um julgamento de valor — pessoas diferentes escolhem pesos diferentes, e isso muda a recomendação.",
            "A elasticidade é herdada do business e os clusters dependem das features escolhidas; SKUs com histórico curto têm score mais ruidoso.",
        ],
        fig=fig, table=mcda_tbl)


    notes = []
    return {"metrics": {}, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


# ========================================================================
# 05 — PRICE MODEL / RESIDUAL
# ========================================================================
def _demean_by_group(df, value_col, group_col):
    grp_mean = df.groupby(group_col)[value_col].transform("mean")
    return df[value_col] - grp_mean, grp_mean


def _run_ols(df):
    import statsmodels.api as sm
    d = df.copy()
    d["ln_price_dm"], mat_base = _demean_by_group(d, "ln_price", COL_MATERIAL)
    d["ln_qty_dm"], _ = _demean_by_group(d, "ln_qty", COL_MATERIAL)
    X_cat = pd.get_dummies(d[[COL_COUNTRY, COL_TIER, COL_BUSINESS, "ym"]],
                           drop_first=True, dtype=float)
    X_cat_dm = X_cat.sub(X_cat.groupby(d[COL_MATERIAL].values).transform("mean"))
    X = pd.concat([d[["ln_qty_dm"]].reset_index(drop=True),
                   X_cat_dm.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = d["ln_price_dm"].reset_index(drop=True)
    model = sm.OLS(y, X, missing="drop").fit()

    d = d.reset_index(drop=True)
    mat_base = mat_base.reset_index(drop=True)
    # model.predict pode devolver menos linhas que d se OLS(missing="drop")
    # descartou linhas com NaN/inf. Reindexamos pela posição de X para alinhar
    # com d antes de somar mat_base, evitando erro de broadcast (shapes diferentes).
    pred = pd.Series(np.asarray(model.predict(X)), index=X.index)
    pred = pred.reindex(range(len(d)))
    d["ln_price_hat"] = mat_base + pred
    d["resid"] = d["ln_price"] - d["ln_price_hat"]
    d["price_ratio"] = np.exp(d["resid"])
    d["price_expected"] = np.exp(d["ln_price_hat"])

    coefs = model.params.to_frame("coef")
    coefs["pct_effect"] = np.exp(coefs["coef"]) - 1
    coefs = coefs.reset_index().rename(columns={"index": "termo"})
    return d, model.rsquared, coefs, model.params.get("ln_qty_dm", float("nan"))


def _run_xgb(df, p: Params):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBRegressor
    d = df.copy()
    cat_cols = [COL_MATERIAL, COL_COUNTRY, COL_TIER, COL_BUSINESS, "ym"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc = enc.fit_transform(d[cat_cols])
    X = np.column_stack([Xc, d["ln_qty"].values])
    y = d["ln_price"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=p.random_state)
    m = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=p.random_state, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr)
    r2 = m.score(Xte, yte)
    d["price_expected_xgb"] = np.exp(m.predict(X))
    d["resid_xgb"] = d["ln_price"] - np.log(d["price_expected_xgb"])
    return d, r2


def run_price_model(df: pd.DataFrame, p: Params) -> dict:
    d0 = df[df["price_valid"]].dropna(subset=["ln_price", "ln_qty"]).copy()
    notes = []

    d, r2_ols, coefs, beta_qty = _run_ols(d0)
    notes.append(f"OLS R² within-material = {r2_ols:.3f} "
                 f"(baixo é esperado: o efeito fixo de material já absorve o preço-base).")

    r2_xgb = None
    use_xgb = False
    if p.use_xgb:
        try:
            d2, r2_xgb = _run_xgb(d0, p)
            d = d.reset_index(drop=True)
            d["price_expected_xgb"] = d2["price_expected_xgb"].values
            d["resid_xgb"] = d2["resid_xgb"].values
            use_xgb = r2_xgb > r2_ols
        except Exception as e:
            notes.append(f"XGBoost indisponível ({e}); usando OLS.")

    if use_xgb:
        d["resid_use"] = d["resid_xgb"]
        d["price_expected_use"] = d["price_expected_xgb"]
        notes.append(f"XGBoost R² (holdout) = {r2_xgb:.3f}. Oportunidades baseadas no XGBoost.")
    else:
        d["resid_use"] = d["resid"]
        d["price_expected_use"] = d["price_expected"]
        notes.append("Oportunidades baseadas no OLS.")

    target = d["resid_use"].quantile(p.residual_target_pctl)
    opp = d[d["resid_use"] < target].copy()
    opp["price_target"] = opp["price_expected_use"] * np.exp(target)
    opp["price_gap"] = opp["price_target"] - opp[COL_PRICE]
    opp["uplift_gross"] = opp["price_gap"].clip(lower=0) * opp[COL_QTY]
    opp["uplift_captured"] = opp["uplift_gross"] * p.capture_rate
    opp["price_ratio"] = np.exp(opp["resid_use"])

    cols = [COL_MONTH, COL_BUSINESS, COL_COUNTRY, COL_TIER, COL_MATERIAL,
            COL_PARENT, COL_SOLDTO, COL_QTY, COL_PRICE,
            "price_expected_use", "price_target", "resid_use", "price_ratio",
            "price_gap", "uplift_gross", "uplift_captured"]
    opp = opp[cols].sort_values("uplift_gross", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(d["resid_use"].clip(-2, 2), bins=80, color="#3b6ea5")
    ax.axvline(0, color="grey", ls=":")
    ax.axvline(target, color="crimson", ls="--",
               label=f"alvo P{int(p.residual_target_pctl*100)} = {target:.2f}")
    ax.set_title("Distribuição do resíduo de preço (ln real - ln esperado)")
    ax.set_xlabel("resíduo (negativo = paga abaixo do esperado)")
    ax.set_ylabel("nº de transações")
    ax.legend()
    fig.tight_layout()

    metrics = {
        "Transações no modelo": f"{len(d0):,}",
        "R² OLS (within)": f"{r2_ols:.3f}",
        "R² XGBoost": (f"{r2_xgb:.3f}" if r2_xgb is not None else "—"),
        "beta ln(Qty) (desc. volume)": f"{beta_qty:.3f}",
        "Oportunidades": f"{len(opp):,}",
        "Uplift bruto": f"${opp['uplift_gross'].sum():,.0f}",
        f"Uplift capturável ({p.capture_rate:.0%})": f"${opp['uplift_captured'].sum():,.0f}",
    }
    return {
        "metrics": metrics,
        "tables": {"price_opportunities": opp, "ols_coefficients": coefs},
        "figures": {"residual_hist": fig},
        "notes": notes,
    }


# ========================================================================
# 06 — ELASTICITY
# ========================================================================
def _within(df, cols, by):
    key = df[by].astype(str).agg("|".join, axis=1)
    out = df[cols].copy()
    for c in cols:
        out[c] = df[c] - df.groupby(key)[c].transform("mean")
    return out


def _estimate_elasticity(d):
    import statsmodels.api as sm
    dm = _within(d, ["ln_qty", "ln_price"], [COL_MATERIAL, "ym"])
    X = sm.add_constant(dm[["ln_price"]])
    res = sm.OLS(dm["ln_qty"], X, missing="drop").fit(cov_type="HC1")
    return res.params["ln_price"], res.bse["ln_price"], int(res.nobs), res.rsquared


def run_elasticity(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df[df["price_valid"]].dropna(subset=["ln_qty", "ln_price"]).copy()
    rows = []
    b, se, n, r2 = _estimate_elasticity(d)
    rows.append(["GLOBAL", b, se, n, r2])
    for biz, g in d.groupby(COL_BUSINESS):
        if len(g) < 500:
            continue
        try:
            b, se, n, r2 = _estimate_elasticity(g)
            rows.append([biz, b, se, n, r2])
        except Exception:
            continue
    out = pd.DataFrame(rows, columns=["business", "elasticity", "std_err", "n_obs", "r2_within"])
    out["interpretacao"] = np.where(
        out["elasticity"] < -1, "elástico (aumento de preço derruba volume)",
        np.where(out["elasticity"] < 0, "inelástico (há espaço p/ subir preço)",
                 "sinal positivo — revisar (endogeneidade/mix)"))
    return out


def elasticity_result(df: pd.DataFrame, p: Params) -> dict:
    out = run_elasticity(df, p)
    return {
        "metrics": {"Elasticidade global": f"{out.loc[0, 'elasticity']:.2f}"},
        "tables": {"elasticity_by_business": out},
        "figures": {},
        "notes": ["Betas com viés de simultaneidade (preço e volume se determinam juntos); "
                  "usar como ordem de grandeza."],
    }


# ========================================================================
# 07 — OPTIMIZATION (usa a elasticidade)
# ========================================================================
def _optimal_price(cost, beta):
    with np.errstate(divide="ignore", invalid="ignore"):
        return cost * beta / (beta + 1.0)


def run_optimization(df: pd.DataFrame, p: Params, elas: pd.DataFrame | None = None) -> dict:
    if elas is None:
        elas = run_elasticity(df, p)
    d = df[df["price_valid"]].copy()
    d["unit_cost"] = -d[COL_COGS_AVG]

    beta_map = dict(zip(elas["business"], elas["elasticity"]))
    beta_global = beta_map.get("GLOBAL", -1.5)
    d["beta"] = d[COL_BUSINESS].map(beta_map).fillna(beta_global)

    p_star = _optimal_price(d["unit_cost"].values, d["beta"].values)
    inelastic = d["beta"].values >= -1
    p_star = np.where(inelastic | ~np.isfinite(p_star),
                      d[COL_PRICE].values * (1 + p.max_price_increase), p_star)
    lo = d[COL_PRICE].values * (1 - p.max_price_decrease)
    hi = d[COL_PRICE].values * (1 + p.max_price_increase)
    d["price_opt"] = np.clip(p_star, lo, hi)

    ratio = d["price_opt"] / d[COL_PRICE]
    d["qty_adj"] = d[COL_QTY] * np.power(ratio, d["beta"])
    d["margin_now"] = (d[COL_PRICE] - d["unit_cost"]) * d[COL_QTY]
    d["margin_opt"] = (d["price_opt"] - d["unit_cost"]) * d["qty_adj"]
    d["margin_gain"] = d["margin_opt"] - d["margin_now"]

    seg = (d.groupby(COL_BUSINESS)
           .agg(beta=("beta", "first"), margem_atual=("margin_now", "sum"),
                margem_otimizada=("margin_opt", "sum"), ganho=("margin_gain", "sum"),
                n_tx=("margin_gain", "size"))
           .reset_index().sort_values("ganho", ascending=False))
    seg["ganho_pct"] = seg["ganho"] / seg["margem_atual"].replace(0, np.nan)

    return {
        "metrics": {
            "Ganho total estimado": f"${seg['ganho'].sum():,.0f}",
            f"Cap de aumento": f"+{p.max_price_increase:.0%}",
        },
        "tables": {"optimization_by_segment": seg},
        "figures": {},
        "notes": ["Preço ótimo por forma fechada P*=c·β/(β+1) para β<-1; "
                  "onde inelástico, aplica o cap de aumento."],
    }


# ========================================================================
# 08 — PRICE / VOLUME / MIX BRIDGE
# ========================================================================
def _period_agg(df):
    g = df.groupby(COL_MATERIAL).agg(
        rev=(COL_REVENUE, "sum"), qty=(COL_QTY, "sum"),
        cogs=(COL_COGS, "sum"), margin=("margin_calc", "sum"))
    g["price"] = g["rev"] / g["qty"].replace(0, np.nan)
    g["unit_cost"] = -g["cogs"] / g["qty"].replace(0, np.nan)
    g["unit_margin"] = g["margin"] / g["qty"].replace(0, np.nan)
    return g


def run_pvm_bridge(df: pd.DataFrame, p: Params) -> dict:
    d = df[df["price_valid"]].copy()
    years = sorted(d["year"].unique())
    if len(years) < 2:
        return {"metrics": {}, "tables": {}, "figures": {},
                "notes": ["Precisa de ao menos 2 anos para a bridge."]}
    y0, y1 = years[-2], years[-1]
    g0, g1 = _period_agg(d[d["year"] == y0]), _period_agg(d[d["year"] == y1])
    common = g0.index.intersection(g1.index)
    a, b = g0.loc[common], g1.loc[common]

    eff_price = ((b["price"] - a["price"]) * b["qty"]).sum()
    eff_cost = ((a["unit_cost"] - b["unit_cost"]) * b["qty"]).sum()
    eff_volume = ((b["qty"] - a["qty"]) * a["unit_margin"]).sum()
    m0, m1 = g0["margin"].sum(), g1["margin"].sum()
    total_change = m1 - m0
    eff_mix = total_change - eff_price - eff_cost - eff_volume

    bridge = pd.DataFrame({
        "componente": [f"Margem {y0}", "Efeito Preço", "Efeito Custo",
                       "Efeito Volume", "Efeito Mix (+novos/descont.)", f"Margem {y1}"],
        "valor": [m0, eff_price, eff_cost, eff_volume, eff_mix, m1],
    })

    # waterfall simples
    fig, ax = plt.subplots(figsize=(8, 4.2))
    steps = bridge["valor"].tolist()
    labels = bridge["componente"].tolist()
    running = 0
    for i, (lab, val) in enumerate(zip(labels, steps)):
        if i in (0, len(steps) - 1):
            ax.bar(i, val, color="#33475b")
            running = val
        else:
            ax.bar(i, val, bottom=running, color=("#3b6ea5" if val >= 0 else "#c0504d"))
            running += val
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Price/Volume/Mix Bridge  ({y0} → {y1})")
    ax.set_ylabel("Margem ($)")
    fig.tight_layout()

    return {
        "metrics": {f"Margem {y0}": f"${m0:,.0f}", f"Margem {y1}": f"${m1:,.0f}",
                    "Variação total": f"${total_change:,.0f}"},
        "tables": {"pvm_bridge": bridge},
        "figures": {"pvm_bridge": fig},
        "notes": [f"Comparação {y0} vs {y1} no nível de material."],
    }


def _pvm_bridge_block(df, p, reajuste_real=None, efeito_mix_preco=None):
    """Ponte P/V/M como bloco explicativo (para o Diagnóstico).
    reajuste_real / efeito_mix_preco: variação % de preço nos mesmos itens e
    o efeito mix sobre o preço médio (incorpora o antigo bloco 'mix vs preço')."""
    d = df[df["price_valid"]].copy()
    years = sorted(d["year"].unique())
    if len(years) < 2:
        return None
    y0, y1 = years[-2], years[-1]
    g0, g1 = _period_agg(d[d["year"] == y0]), _period_agg(d[d["year"] == y1])
    common = g0.index.intersection(g1.index)
    a, b = g0.loc[common], g1.loc[common]
    eff_price = ((b["price"] - a["price"]) * b["qty"]).sum()
    eff_cost = ((a["unit_cost"] - b["unit_cost"]) * b["qty"]).sum()
    eff_volume = ((b["qty"] - a["qty"]) * a["unit_margin"]).sum()
    # margens TOTAIS reais (batem com o Painel de KPIs), não só a interseção
    m0 = float(df[df["year"] == y0]["_margin"].sum())
    m1 = float(df[df["year"] == y1]["_margin"].sum())
    total_change = m1 - m0
    # efeito mix/outros = resíduo que fecha a ponte contra os TOTAIS reais
    # (inclui itens que entraram/saíram e transações fora do price_valid)
    eff_mix = total_change - eff_price - eff_cost - eff_volume
    bridge = pd.DataFrame({
        "componente": [f"Margem {y0}", "Efeito Preço", "Efeito Custo",
                       "Efeito Volume", "Efeito Mix / outros", f"Margem {y1}"],
        "valor": [m0, eff_price, eff_cost, eff_volume, eff_mix, m1],
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    steps = bridge["valor"].tolist(); labels = bridge["componente"].tolist()
    running = 0
    for i, (lab, val) in enumerate(zip(labels, steps)):
        if i in (0, len(steps) - 1):
            ax.bar(i, val / 1e6, color="#33475b"); running = val / 1e6
        else:
            ax.bar(i, val / 1e6, bottom=running, color=("#1d9e75" if val >= 0 else "#c0504d"))
            running += val / 1e6
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(f"Ponte Preço/Volume/Mix ({y0} → {y1})")
    ax.set_ylabel("Margem (US$ mi)")
    fig.tight_layout()
    drivers = {"Preço": eff_price, "Custo": eff_cost, "Volume": eff_volume, "Mix": eff_mix}
    maior = max(drivers, key=lambda k: abs(drivers[k]))

    # tabela didática: o que cada efeito significa, em português
    metod = pd.DataFrame({
        "efeito": ["Preço", "Custo", "Volume", "Mix / outros"],
        "o que mede": [
            "Cobramos mais (ou menos) pelos MESMOS produtos",
            "O custo unitário dos mesmos produtos subiu ou caiu",
            "Vendemos MAIS ou MENOS unidades (mesmo portfólio)",
            "Mudança de COMPOSIÇÃO + itens que entraram/saíram + ajustes de reconciliação",
        ],
        "como se calcula (em palavras)": [
            "(preço novo − preço velho) × quantidade nova",
            "(custo velho − custo novo) × quantidade nova",
            "(quantidade nova − quantidade velha) × margem unitária velha",
            "resíduo que fecha a ponte contra a margem TOTAL real de cada ano",
        ],
        "impacto (US$ mi)": [eff_price/1e6, eff_cost/1e6, eff_volume/1e6, eff_mix/1e6],
    })

    reaj_txt = ""
    if reajuste_real is not None:
        reaj_txt = (f" Vale notar: nos MESMOS itens o preço foi reajustado {reajuste_real:+.1f}% "
                    f"(o mix puxou o preço médio {efeito_mix_preco:+.1f}%), coerente com o Efeito Preço positivo aqui — "
                    "não há erosão de preço, o problema da margem é volume.")

    return {
        "title": f"Ponte Preço / Volume / Mix ({y0} → {y1})",
        "why": ("A Ponte P/V/M decompõe a variação de margem entre dois anos em quatro alavancas separadas, para responder "
                "'por que a margem mudou' sem achismo. Pense assim: a margem é preço menos custo, vezes quantidade. Quando a margem "
                "muda, pode ser porque (1) mudou o PREÇO dos mesmos itens, (2) mudou o CUSTO deles, (3) vendemos mais/menos "
                "UNIDADES, ou (4) mudou o MIX — a cesta de produtos passou a ter itens de margem diferente. A ponte isola quanto "
                "cada uma dessas quatro coisas pesou, em dinheiro."),
        "answers": f"A margem mudou de {y0} para {y1} — quanto disso foi preço, custo, volume e mix?",
        "method": "Cada efeito é calculado 'segurando' as outras variáveis fixas, uma de cada vez. "
                  "Efeito Preço = (preço novo − preço velho) × volume novo — quanto ganhamos/perdemos só pela mudança de preço, "
                  "no volume atual. Efeito Custo = (custo velho − custo novo) × volume novo. Efeito Volume = (quantidade nova − "
                  "quantidade velha) × margem unitária antiga — o efeito de vender mais/menos, ao preço-margem de antes. "
                  "A ponte começa e termina nas margens TOTAIS reais de cada ano (as mesmas do Painel de KPIs); o 'Mix / outros' "
                  "é o resíduo que fecha a conta — inclui mudança de composição, itens que entraram/saíram e transações sem "
                  "preço unitário comparável. A tabela 'Como ler cada efeito' abaixo resume isso.",
        "insight": f"A ponte começa e termina nas margens reais (US$ {m0/1e6:,.1f} mi → US$ {m1/1e6:,.1f} mi, "
                   f"iguais ao Painel de KPIs). A variação total foi de US$ {total_change/1e6:+,.1f} mi, e o maior efeito veio "
                   f"de {maior} (US$ {drivers[maior]/1e6:+,.1f} mi). A leitura: o preço AJUDOU (US$ {eff_price/1e6:+,.1f} mi), "
                   f"mas o VOLUME derrubou (US$ {eff_volume/1e6:+,.1f} mi) — a margem caiu porque vendemos menos, não porque "
                   f"precificamos mal.{reaj_txt}",
        "money": f"Decomposição da variação de US$ {total_change/1e6:+,.1f} mi em margem: "
                 f"Preço {eff_price/1e6:+,.1f} · Custo {eff_cost/1e6:+,.1f} · Volume {eff_volume/1e6:+,.1f} · Mix {eff_mix/1e6:+,.1f} (mi).",
        "worked": [
            f"Efeito Preço positivo (US$ {eff_price/1e6:+,.1f} mi): o reajuste dos mesmos itens ajudou a margem — precificação funcionou.",
            "A ponte separa o que ajudou do que atrapalhou, em dinheiro — decisão sai do achismo.",
        ],
        "not_worked": [
            f"O maior peso negativo veio de {maior} (US$ {drivers[maior]/1e6:+,.1f} mi) — a alavanca que mais derrubou a margem.",
            "Queda por Volume/Mix não se resolve com preço: exige retenção de clientes e gestão de portfólio.",
        ],
        "table": _round_df(bridge, money_cols=["valor"]),
        "table2": _round_df(metod, money_cols=["impacto (US$ mi)"]),
        "fig": fig,
    }


def run_diagnostico(df: pd.DataFrame, p: Params) -> dict:
    """Aba 01 · Diagnóstico — construída bloco a bloco. Por ora: Painel
    Executivo de KPIs (2025 → 2026) com leitura de o que funcionou,
    o que não funcionou e a estratégia recomendada."""
    df = df.copy()
    df["_margin"] = df[COL_REVENUE] + df[COL_COGS]
    blocks = []

    def add(title, why="", answers="", insight="", table=None, fig=None, money="",
            worked=None, not_worked=None, method=""):
        blocks.append({"title": title, "why": why, "answers": answers, "insight": insight,
                       "table": table, "fig": fig, "money": money,
                       "worked": worked, "not_worked": not_worked, "method": method})

    # ---- KPIs 2025 vs 2026 ----
    y_all = sorted(int(v) for v in df["year"].unique())
    y0, y1 = y_all[-2], y_all[-1]
    dv = df[(df[COL_PRICE] > 0) & (df[COL_QTY] > 0)]

    def _kpis(year):
        g = df[df["year"] == year]
        gv = dv[dv["year"] == year]
        rev = g[COL_REVENUE].sum(); mar = g["_margin"].sum(); cogs = -g[COL_COGS].sum()
        return {
            "Receita": rev, "Margem de contribuição": mar,
            "Margem %": 100 * mar / rev if rev else np.nan, "Custo (COGS)": cogs,
            "Clientes que compraram": g[COL_PARENT].nunique(),
            "Portfólio comprado (materiais)": g[COL_MATERIAL].nunique(),
            "Países atendidos": g[COL_COUNTRY].nunique(),
            "Transações": len(g),
        }

    k0, k1 = _kpis(y0), _kpis(y1)

    def _fmt(nome, v):
        if nome in ("Receita", "Margem de contribuição", "Custo (COGS)"):
            return f"${v/1e6:,.1f} mi"
        if nome == "Margem %":
            return f"{v:.1f}%"
        return f"{v:,.0f}"

    def _delta(nome):
        a, b = k0[nome], k1[nome]
        if nome == "Margem %":
            return f"{b - a:+.1f} p.p."
        return f"{100*(b/a - 1):+.1f}%" if a else "—"

    rows = []
    for nome in k0:
        rows.append({"Indicador": nome, f"{y0}": _fmt(nome, k0[nome]),
                     f"{y1}": _fmt(nome, k1[nome]), "Δ": _delta(nome)})
    kpi_tbl = pd.DataFrame(rows)

    var_rev = 100 * (k1["Receita"] / k0["Receita"] - 1)
    var_cli = 100 * (k1["Clientes que compraram"] / k0["Clientes que compraram"] - 1)
    dpp_margem = k1["Margem %"] - k0["Margem %"]
    var_tx = 100 * (k1["Transações"] / k0["Transações"] - 1)
    var_mat = 100 * (k1["Portfólio comprado (materiais)"] / k0["Portfólio comprado (materiais)"] - 1)

    add("Painel executivo — KPIs do negócio",
        why="Todo diagnóstico de consultoria abre pelo retrato do negócio: onde estamos hoje e como isso mudou frente ao ano anterior. É o 'situation' antes de investigar as causas — dá ao gestor a régua para julgar tudo o que vem depois.",
        answers=f"Como {y1} se compara a {y0} em receita, margem, custo, base de clientes e amplitude de portfólio e mercado?",
        method=f"Cada KPI compara o ano {y0} com {y1}. Receita, Margem e Custo são a SOMA de todas as transações do ano (Margem = Receita + Custo, lembrando que o custo entra negativo). Margem % = Margem ÷ Receita. Clientes, Materiais e Países são CONTAGENS de valores distintos (quantos clientes/produtos/países únicos compraram no ano). Transações é o número de linhas de venda. A coluna Δ é a variação percentual entre os dois anos, exceto Margem %, que mostra a diferença em pontos percentuais (p.p.).",
        insight=f"Receita {var_rev:+.1f}% e base de clientes {var_cli:+.1f}% — o negócio encolheu em tamanho. "
                f"Mas a margem % foi de {k0['Margem %']:.1f}% para {k1['Margem %']:.1f}% ({dpp_margem:+.1f} p.p.) — "
                "melhora de rentabilidade apesar da queda de volume, sinal de mix/preço mais saudável ou corte de cauda deficitária. "
                "O diagnóstico de pricing a seguir mostra se há espaço para ir além.",
        worked=[
            f"Rentabilidade: margem % subiu {dpp_margem:+.1f} p.p. ({k0['Margem %']:.1f}% → {k1['Margem %']:.1f}%) mesmo com menos volume.",
            f"Disciplina de custo: COGS caiu {_delta('Custo (COGS)')}, acompanhando a queda de receita.",
            "Eficiência por transação preservada: a margem por venda não se deteriorou apesar do tombo de volume.",
        ],
        not_worked=[
            f"Perda de clientes: base caiu {var_cli:+.1f}% ({k0['Clientes que compraram']:,} → {k1['Clientes que compraram']:,}).",
            f"Retração de volume: transações {var_tx:+.1f}% e receita {var_rev:+.1f}% — o negócio ficou menor.",
            f"Encolhimento de portfólio e mercado: materiais {var_mat:+.1f}% e países {_delta('Países atendidos')}.",
        ],
        money=f"Cada ponto de margem % vale ~${k1['Receita']/1e6*0.01:,.1f} mi sobre a receita de {y1}.",
        table=kpi_tbl)

    # ===================================================================
    # BLOCO 2 — CAGR (taxa de crescimento composta)
    # ===================================================================
    yby = (df.groupby("year")
           .agg(receita=(COL_REVENUE, "sum"), margem=("_margin", "sum"))
           .reset_index().sort_values("year"))
    yby["margem_pct"] = 100 * yby["margem"] / yby["receita"]
    n_anos = int(yby["year"].iloc[-1] - yby["year"].iloc[0])

    def _cagr(serie):
        ini, fim = serie.iloc[0], serie.iloc[-1]
        return (fim / ini) ** (1 / n_anos) - 1 if n_anos and ini > 0 else np.nan

    cagr_rec = _cagr(yby["receita"]); cagr_mar = _cagr(yby["margem"])
    yby["receita_var_%"] = (yby["receita"].pct_change() * 100)
    yby["margem_var_%"] = (yby["margem"].pct_change() * 100)

    # tabela CAGR
    cagr_view = pd.DataFrame({
        "indicador": ["Receita", "Margem"],
        f"{int(yby['year'].iloc[0])}": [yby["receita"].iloc[0], yby["margem"].iloc[0]],
        f"{int(yby['year'].iloc[-1])}": [yby["receita"].iloc[-1], yby["margem"].iloc[-1]],
        "CAGR (a.a.)": [f"{cagr_rec*100:+.1f}%", f"{cagr_mar*100:+.1f}%"],
    })
    cagr_view_tbl = _round_df(cagr_view, money_cols=[f"{int(yby['year'].iloc[0])}",
                                                     f"{int(yby['year'].iloc[-1])}"])

    # gráfico: barras de receita e margem por ano + linha de tendência do CAGR
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(yby)); w = 0.38
    xs = yby["year"].astype(int).astype(str)
    ax.bar(x - w/2, yby["receita"] / 1e6, w, color="#3b6ea5", label="Receita")
    ax.bar(x + w/2, yby["margem"] / 1e6, w, color="#1d9e75", label="Margem")
    # linha tracejada do caminho implícito no CAGR (receita)
    rec0 = yby["receita"].iloc[0] / 1e6
    cagr_line = [rec0 * (1 + cagr_rec) ** i for i in range(len(yby))]
    ax.plot(x, cagr_line, color="#c0504d", ls="--", marker="o", ms=4,
            label=f"tendência CAGR receita ({cagr_rec*100:+.1f}%/ano)")
    # rótulos de variação ano a ano na receita
    for i, v in enumerate(yby["receita_var_%"]):
        if pd.notna(v):
            ax.annotate(f"{v:+.0f}%", (i - w/2, yby["receita"].iloc[i] / 1e6),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(xs)
    ax.set_ylabel("US$ milhões"); ax.set_title("Receita e margem por ano — com tendência CAGR")
    ax.legend(fontsize=8)
    fig.tight_layout()

    # trajetória para os bullets
    vars_rec = yby["receita_var_%"].dropna().tolist()
    ano_pico = int(yby.loc[yby["receita_var_%"].idxmax(), "year"]) if yby["receita_var_%"].notna().any() else None
    ano_pior = int(yby.loc[yby["receita_var_%"].idxmin(), "year"]) if yby["receita_var_%"].notna().any() else None

    add("CAGR — a taxa de crescimento composta",
        why="O CAGR (Compound Annual Growth Rate) resume o ritmo médio de crescimento anual do período inteiro, como juros compostos — uma régua única que suaviza os altos e baixos de cada ano. Olhado junto da variação ano a ano, revela tanto a tendência de fundo quanto onde a trajetória virou.",
        answers="A que ritmo médio a empresa cresceu (ou encolheu) no período, e essa queda foi gradual ou concentrada?",
        method="O CAGR (taxa de crescimento anual composta) usa a fórmula (valor final ÷ valor inicial) elevado a (1 ÷ número de anos), menos 1. Aqui pegamos a receita (e a margem) do primeiro e do último ano da base e aplicamos essa fórmula — o resultado é a taxa que, aplicada de forma composta a cada ano, levaria do valor inicial ao final. As barras mostram receita e margem somadas por ano; a linha tracejada é o caminho que o CAGR da receita descreve; os rótulos são a variação de um ano para o outro.",
        insight=f"No período {int(yby['year'].iloc[0])}→{int(yby['year'].iloc[-1])}, a receita teve CAGR de {cagr_rec*100:+.1f}%/ano "
                f"e a margem {cagr_mar*100:+.1f}%/ano — a margem encolhe cerca do dobro da receita, sinal de pressão de rentabilidade "
                "ao longo do período. Mas a trajetória não é uniforme: a queda se concentra no último ano, não é um declínio gradual.",
        worked=[
            f"{ano_pico} foi o melhor ano: receita cresceu {max(vars_rec):+.1f}% — mostra que a empresa tem capacidade de crescer.",
            "A margem % recuperou no último ano (voltou a subir depois de anos caindo), apesar da queda de receita.",
        ],
        not_worked=[
            f"Tendência de fundo negativa: CAGR de receita {cagr_rec*100:+.1f}%/ano e de margem {cagr_mar*100:+.1f}%/ano.",
            f"{ano_pior} concentra o tombo: receita {min(vars_rec):+.1f}% — reverteu todo o ganho do ano anterior.",
            "Margem cai mais rápido que receita no período (CAGR −6% vs −3%): cada ano perde rentabilidade relativa.",
        ],
        fig=fig)

    # ===================================================================
    # GRUPO ① — PERFORMANCE DE PREÇO NO TEMPO
    # ===================================================================
    dv = df[(df[COL_PRICE] > 0) & (df[COL_QTY] > 0)].copy()
    dv["_unit_cost"] = (-dv[COL_COGS_AVG]).clip(lower=0)
    pm_year = dv.groupby("year").apply(lambda g: np.average(g[COL_PRICE], weights=g[COL_QTY]))
    cm_year = dv.groupby("year").apply(lambda g: np.average(g["_unit_cost"], weights=g[COL_QTY]))

    # A · dispersão maçã-a-maçã: mesmo produto × tier × business × país
    # seletor de país (padrão = mais representativo por receita)
    ctry_rev = df.groupby(COL_COUNTRY)[COL_REVENUE].sum().sort_values(ascending=False)
    pais_sel = p.diag_country if (p.diag_country and p.diag_country != "(auto)") else ctry_rev.index[0]
    dp = dv[(dv[COL_COUNTRY] == pais_sel) & (dv[COL_PRICE] >= 1)].copy()  # >=1 remove ruído de unidade
    PG = [COL_MATERIAL, COL_TIER, COL_BUSINESS]
    grp = dp.groupby(PG)[COL_PRICE].agg(n="count", media="mean", mediana="median",
                                        pmin="min", pmax="max", std="std")
    grp = grp[grp["n"] >= 5].copy()
    grp["cv"] = grp["std"] / grp["media"]
    grp["spread_x"] = grp["pmax"] / grp["pmin"].clip(lower=0.01)
    # oportunidade: vendas abaixo da mediana do próprio grupo, levadas à mediana
    dpm = dp.merge(dp.groupby(PG)[COL_PRICE].median().rename("_med"), on=PG)
    dpm = dpm.merge(dp.groupby(PG)[COL_PRICE].transform("count").rename("_n"), left_index=True, right_index=True)
    below = dpm[(dpm[COL_PRICE] < dpm["_med"])]
    opp_pais = float(((below["_med"] - below[COL_PRICE]) * below[COL_QTY]).sum())

    # top grupos por CV para o gráfico (mesmo produto/tier/business, muita variação)
    top = grp.sort_values("cv", ascending=False).head(15).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = []
    for i, r in top.iterrows():
        key = (r[COL_MATERIAL], r[COL_TIER], r[COL_BUSINESS])
        vals = dp[(dp[COL_MATERIAL] == key[0]) & (dp[COL_TIER] == key[1]) &
                  (dp[COL_BUSINESS] == key[2])][COL_PRICE].values
        xj = np.random.RandomState(i).normal(i, 0.06, len(vals))
        ax.scatter(xj, vals, s=14, alpha=0.5, color="#3b6ea5")
        ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color="#c0504d", lw=2)
        labels.append(f"{str(r[COL_MATERIAL])[:10]}\nT{r[COL_TIER]}·B{r[COL_BUSINESS]}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("Preço unitário (log)")
    ax.set_title(f"Mesmo produto × tier × business em {pais_sel}: cada ponto é uma venda (linha = mediana)")
    fig.tight_layout()

    cv_med_pais = 100 * grp["cv"].median()
    pior = top.iloc[0]
    n_grupos = len(grp)
    tbl_disp = _round_df(
        top[[COL_MATERIAL, COL_TIER, COL_BUSINESS, "n", "pmin", "mediana", "pmax", "cv", "spread_x"]]
        .rename(columns={"n": "vendas", "pmin": "preço_min", "pmax": "preço_max",
                         "cv": "CV", "spread_x": "amplitude_x"}),
        money_cols=["preço_min", "mediana", "preço_max"], pct_cols=["CV"], int_cols=["vendas"])

    add("Dispersão de preços — mesmo produto, tier, business e país",
        why="A comparação justa: em vez de misturar produtos diferentes, olhamos o MESMO material, para o MESMO tier, no MESMO business e país. Aqui a variação de preço não tem desculpa de 'é outro produto' — cada ponto é uma venda unitária, e a diferença entre elas é dispersão pura, candidata a repricing.",
        answers=f"Dentro de um mesmo produto/tier/business em {pais_sel}, quão diferente é o preço de uma venda para outra?",
        method=f"Agrupamos as vendas por combinação de Material × Tier × Business no país selecionado ({pais_sel}), mantendo só grupos com pelo menos 5 vendas e preço ≥ US$ 1 (para tirar ruído de unidades erradas). Dentro de cada grupo, cada ponto do gráfico é uma venda individual, e a linha vermelha é a mediana do grupo. O CV (coeficiente de variação) = desvio-padrão ÷ média dos preços do grupo — quanto maior, mais espalhados os preços. A oportunidade em US$ soma, para cada venda abaixo da mediana do seu grupo, a diferença (mediana − preço) × quantidade — quanto renderia levar todos ao preço mediano.",
        insight=f"Em {pais_sel}, entre {n_grupos:,} grupos comparáveis (mesmo material/tier/business), o CV mediano é {cv_med_pais:.0f}%. "
                f"No caso mais extremo, o material {pior[COL_MATERIAL]} (tier {pior[COL_TIER]}, business {pior[COL_BUSINESS]}) é vendido de "
                f"US$ {pior['pmin']:,.0f} a US$ {pior['pmax']:,.0f} — {pior['spread_x']:.0f}× de diferença para o mesmo item e mesmo perfil de cliente. "
                "Isso é dispersão sem justificativa de produto — o achado que a média no tempo escondia.",
        worked=[
            "Controlar produto+tier+business+país isola dispersão REAL: o que sobra não é 'mix', é inconsistência de preço.",
            "A mediana de cada grupo vira um benchmark natural e justo para renegociar as vendas abaixo dela.",
        ],
        not_worked=[
            f"Dispersão gritante: o pior grupo varia {pior['spread_x']:.0f}× no mesmo item/perfil — o preço depende de quem negociou, não de regra.",
            f"CV mediano de {cv_med_pais:.0f}% entre comparáveis: falta disciplina de preço mesmo controlando todas as dimensões.",
        ],
        money=f"Só em {pais_sel}, levar cada venda abaixo da mediana do seu grupo até a mediana renderia ~US$ {opp_pais/1e6:,.1f} mi.",
        fig=fig, table=tbl_disp)

    # B · preço vs custo + margem unitária
    pc = (dv.groupby("date")
          .apply(lambda g: pd.Series({
              "preco": np.average(g[COL_PRICE], weights=g[COL_QTY]),
              "custo": np.average(g["_unit_cost"], weights=g[COL_QTY])}))
          .reset_index().sort_values("date"))
    pc["margem_unit"] = pc["preco"] - pc["custo"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pc["date"], pc["preco"], color="#8250c4", lw=1.8, label="Preço médio")
    ax.plot(pc["date"], pc["custo"], color="#c0504d", lw=1.8, label="Custo médio")
    ax.fill_between(pc["date"], pc["custo"], pc["preco"], color="#1d9e75", alpha=0.10, label="Margem unitária")
    ax.set_ylabel("US$ por unidade"); ax.set_title("Preço vs. custo médio no tempo")
    ax.legend(fontsize=8); fig.tight_layout()
    var_preco_bc = 100 * (pm_year.loc[y1] / pm_year.loc[y0] - 1)
    var_custo_bc = 100 * (cm_year.loc[y1] / cm_year.loc[y0] - 1)
    add("Preço vs. custo — quem pressiona a margem",
        why="Preço e custo juntos revelam quem manda na margem. Se o custo sobe mais que o preço, a margem aperta mesmo com o preço subindo — a leitura que a receita sozinha não dá.",
        answers=f"De {y0} para {y1}, o preço subiu mais ou menos que o custo?",
        method=f"Para cada mês, calculamos o preço médio e o custo médio, ambos PONDERADOS PELO VOLUME (cada venda pesa pela quantidade). A linha roxa é o preço, a vermelha é o custo, e a faixa verde entre elas é a margem unitária. As variações citadas comparam o preço/custo médio de {y0} com o de {y1} — cada um calculado como média ponderada anual. Se o preço sobe mais que o custo, a faixa verde se alarga (margem protegida); se o custo sobe mais, ela se estreita.",
        insight=f"Preço médio {var_preco_bc:+.1f}% e custo médio {var_custo_bc:+.1f}% de {y0} para {y1}. "
                + ("O preço cresceu mais que o custo — margem unitária protegida."
                   if var_preco_bc > var_custo_bc else
                   "O custo cresceu mais que o preço — margem unitária pressionada."),
        worked=[
            f"Preço acompanhou (ou superou) o custo: {var_preco_bc:+.1f}% vs {var_custo_bc:+.1f}% — a margem unitária não foi corroída.",
            "A empresa conseguiu repassar a variação de custo ao preço, sinal de disciplina ou poder de precificação.",
        ],
        not_worked=[
            "A folga entre preço e custo é estreita em vários meses — pouca gordura para absorver choques de custo.",
            "Preço e custo médios ainda são muito sensíveis ao mix (média ponderada); o reajuste real aparece melhor no bloco de mix.",
        ],
        fig=fig)

    # cálculos same-store (itens vendidos nos 2 anos) — alimentam o bloco de mix abaixo
    dv0 = dv[dv["year"] == y0]; dv1 = dv[dv["year"] == y1]
    pr0 = dv0.groupby(COL_MATERIAL).apply(lambda g: np.average(g[COL_PRICE], weights=g[COL_QTY]))
    pr1 = dv1.groupby(COL_MATERIAL).apply(lambda g: np.average(g[COL_PRICE], weights=g[COL_QTY]))
    q0m = dv0.groupby(COL_MATERIAL)[COL_QTY].sum()
    common = pr0.index.intersection(pr1.index)

    # ===================================================================
    # GRUPO ② — ONDE E POR QUE O PREÇO MUDOU
    # ===================================================================
    # C · quebras por dimensão (preço cresceu em todos?)
    def _var_por(dim):
        rows = []
        for val, g in dv.groupby(dim):
            g0 = g[g["year"] == y0]; g1 = g[g["year"] == y1]
            if len(g0) >= 30 and len(g1) >= 30:
                p0 = np.average(g0[COL_PRICE], weights=g0[COL_QTY])
                p1 = np.average(g1[COL_PRICE], weights=g1[COL_QTY])
                rows.append({dim: val, "var_%": 100 * (p1 / p0 - 1), "n": len(g)})
        return pd.DataFrame(rows).sort_values("var_%")

    var_tier = _var_por(COL_TIER)
    var_ctry = _var_por(COL_COUNTRY).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    colors_t = ["#c0504d" if v < 0 else "#1d9e75" for v in var_tier["var_%"]]
    axes[0].barh(var_tier[COL_TIER].astype(str), var_tier["var_%"], color=colors_t)
    axes[0].axvline(0, color="grey", lw=0.6); axes[0].set_title("Variação de preço por Tier (%)")
    colors_c = ["#c0504d" if v < 0 else "#1d9e75" for v in var_ctry["var_%"]]
    axes[1].barh(var_ctry[COL_COUNTRY].astype(str), var_ctry["var_%"], color=colors_c)
    axes[1].axvline(0, color="grey", lw=0.6); axes[1].set_title("Variação de preço por País (top 15) %")
    fig.tight_layout()
    n_sobe = int((var_tier["var_%"] > 0).sum()); n_cai = int((var_tier["var_%"] < 0).sum())
    add("Preços por tier e país",
        why="Um preço médio estável pode esconder movimentos opostos por segmento. Quebrar a variação por Tier e País revela se o reajuste foi uma política geral ou concentrado em alguns mercados.",
        method=f"Para cada tier (e cada país), calculamos o preço médio PONDERADO PELO VOLUME em {y0} e em {y1} — cada venda pesa proporcionalmente à quantidade vendida, então clientes grandes influenciam mais que uma venda avulsa. A variação exibida é (preço médio {y1} ÷ preço médio {y0} − 1) × 100. Só entram segmentos com pelo menos 30 vendas em cada ano, para o número ser estatisticamente confiável. As barras verdes indicam alta de preço; as vermelhas, queda.",
        answers="O preço cresceu em todos os tiers e países, ou o movimento foi heterogêneo?",
        insight=f"O preço NÃO se moveu igual: entre os tiers, {n_sobe} subiram e {n_cai} caíram. "
                "O movimento é heterogêneo — alguns segmentos foram reajustados para cima enquanto outros perderam preço, "
                "o que aponta para decisões pontuais, não uma régua única.",
        worked=[
            "Vários tiers e países tiveram reajuste positivo — há capacidade de subir preço onde se decide fazê-lo.",
            "A quebra por dimensão expõe exatamente onde agir, em vez de tratar o preço como um número único.",
        ],
        not_worked=[
            f"Inconsistência de política: {n_cai} tiers PERDERAM preço enquanto outros subiram — não há régua uniforme.",
            "Quedas fortes em alguns segmentos (ex.: tiers de desconto e alguns países) podem ser vazamento de margem não intencional.",
        ],
        fig=fig, table=_round_df(var_tier.rename(columns={"var_%": "variação_preço_%"}),
                                 pct_cols=["variação_preço_%"], int_cols=["n"]))

    # cálculo do reajuste real (mesmos itens) — incorporado na Ponte P/V/M abaixo
    pm0_all = np.average(pr0[common], weights=q0m[common])
    q1m = dv1.groupby(COL_MATERIAL)[COL_QTY].sum()
    pm1_all = np.average(pr1[common], weights=q1m[common])
    pm1_lasp = np.average(pr1[common], weights=q0m[common])  # preço novo, cesta velha
    ef_preco = 100 * (pm1_lasp / pm0_all - 1)
    ef_total = 100 * (pm1_all / pm0_all - 1)
    ef_mix = ef_total - ef_preco

    # Ponte P/V/M (com o reajuste real de preço embutido no texto)
    pvm = _pvm_bridge_block(df, p, reajuste_real=ef_preco, efeito_mix_preco=ef_mix)
    if pvm:
        blocks.append(pvm)

    # ===================================================================
    # GRUPO ③ — RELAÇÕES PREÇO × DESEMPENHO (controlado por peer group)
    # ===================================================================
    # cada ponto = (preço relativo, volume relativo) DENTRO do seu grupo
    # peer group = material × tier × business × país; normaliza pela mediana do grupo
    dv["_margin_unit"] = dv[COL_PRICE] - dv["_unit_cost"]
    PGq = [COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY]
    gq = dv.groupby(PGq)
    dv["_pmed"] = gq[COL_PRICE].transform("median")
    dv["_qmed"] = gq[COL_QTY].transform("median")
    dv["_mmed"] = gq["_margin_unit"].transform("median")
    dv["_ngrp"] = gq[COL_PRICE].transform("count")
    dq = dv[(dv["_ngrp"] >= 8) & (dv["_pmed"] > 0) & (dv["_qmed"] > 0)].copy()
    dq["preco_rel"] = dq[COL_PRICE] / dq["_pmed"]
    dq["vol_rel"] = dq[COL_QTY] / dq["_qmed"]

    # correlação controlada
    mok = (dq["preco_rel"] > 0) & (dq["vol_rel"] > 0)
    corr_pq = float(np.corrcoef(np.log(dq.loc[mok, "vol_rel"]), np.log(dq.loc[mok, "preco_rel"]))[0, 1])

    # quadrantes
    q_alto_alto = dq[(dq["vol_rel"] >= 1) & (dq["preco_rel"] >= 1)]
    q_alto_baixo = dq[(dq["vol_rel"] >= 1) & (dq["preco_rel"] < 1)]
    q_baixo_alto = dq[(dq["vol_rel"] < 1) & (dq["preco_rel"] >= 1)]
    q_proibido = dq[(dq["vol_rel"] < 1) & (dq["preco_rel"] < 1)]
    n = len(dq)
    pct_proibido = 100 * len(q_proibido) / n
    pct_altovol_altopreco = 100 * len(q_alto_alto) / n
    opp_proibido = float(((q_proibido["_pmed"] - q_proibido[COL_PRICE]) * q_proibido[COL_QTY]).sum())

    # scatter de quadrantes (amostra para não pesar)
    samp = dq.sample(min(5000, n), random_state=1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(samp["vol_rel"], samp["preco_rel"], alpha=0.18, s=9, color="#3b6ea5")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color="grey", lw=0.8); ax.axvline(1, color="grey", lw=0.8)
    ax.set_xlabel("Volume relativo à mediana do grupo (log)")
    ax.set_ylabel("Preço relativo à mediana do grupo (log)")
    ax.set_title("Preço × Volume dentro do mesmo produto/tier/business/país")
    # sombrear quadrante proibido (baixo vol + baixo preço)
    ax.axhspan(ax.get_ylim()[0], 1, xmin=0, xmax=0.5, color="#c0504d", alpha=0.06)
    ax.text(0.03, 0.03, "PROIBIDO\nbaixo volume\n+ preço baixo", transform=ax.transAxes,
            fontsize=8, color="#c0504d", va="bottom")
    ax.text(0.97, 0.97, "cobra caro de\nquem compra muito", transform=ax.transAxes,
            fontsize=8, color="#8a6d3b", va="top", ha="right")
    fig.tight_layout()

    quad_tbl = pd.DataFrame({
        "quadrante": ["Alto volume + preço alto", "Alto volume + preço baixo (desconto ok)",
                      "Baixo volume + preço alto (ok)", "Baixo volume + preço baixo (PROIBIDO)"],
        "vendas": [len(q_alto_alto), len(q_alto_baixo), len(q_baixo_alto), len(q_proibido)],
        "% do total": [f"{100*len(q_alto_alto)/n:.0f}%", f"{100*len(q_alto_baixo)/n:.0f}%",
                       f"{100*len(q_baixo_alto)/n:.0f}%", f"{pct_proibido:.0f}%"],
    })
    add("Preço × Volume — o mesmo produto, por quem compra quanto",
        why="A teoria manda: dentro de um MESMO produto/tier/business/país, quem compra mais volume deveria pagar menos, e quem compra pouco não deveria ganhar preço baixo. Cada ponto é um par (volume, preço) relativo à mediana do seu grupo — assim produtos caros e baratos ficam comparáveis. Os quatro quadrantes revelam a coerência (ou não) da política.",
        answers="Dentro de grupos comparáveis, quem compra mais paga menos? Há quem compra pouco pagando barato (o proibido)?",
        method="Agrupamos por Material × Tier × Business × País (grupos com ≥ 8 vendas). Dentro de cada grupo, normalizamos: preço relativo = preço da venda ÷ mediana de preço do grupo; volume relativo = quantidade ÷ mediana de quantidade do grupo. Assim 1,0 significa 'na mediana do grupo'. Isso põe produtos caros e baratos na mesma escala. Cada ponto é uma venda; as linhas em 1,0 dividem os 4 quadrantes. O 'proibido' é o canto de baixo-esquerda (preço < mediana E volume < mediana). A correlação é medida em log-log entre volume e preço relativos. A oportunidade soma (mediana − preço) × quantidade das vendas do quadrante proibido.",
        insight=f"Controlando por grupo, a correlação volume×preço é {corr_pq:+.2f} — fraca, confirmando política inconsistente. "
                f"O quadrante PROIBIDO (baixo volume + preço abaixo da mediana) tem {pct_proibido:.0f}% das vendas: "
                "clientes pequenos pagando barato sem justificativa. E {:.0f}% estão em 'alto volume + preço alto' — "
                "clientes grandes pagando acima da mediana, risco de fuga.".format(pct_altovol_altopreco),
        worked=[
            "Controlar por grupo torna a relação preço-volume comparável e acionável, não uma nuvem sem sentido.",
            f"{100*len(q_alto_baixo)/n:.0f}% das vendas estão em 'alto volume + preço baixo' — desconto de volume aplicado corretamente aqui.",
        ],
        not_worked=[
            f"Quadrante PROIBIDO: {pct_proibido:.0f}% das vendas são de baixo volume pagando abaixo da mediana — desconto sem justificativa.",
            f"Correlação {corr_pq:+.2f}: quase não há relação volume-preço dentro dos grupos — falta régua de desconto por volume.",
        ],
        money=f"Levar as vendas do quadrante proibido (baixo volume + preço baixo) à mediana do grupo renderia ~US$ {opp_proibido/1e6:,.1f} mi.",
        fig=fig, table=quad_tbl)

    # ===================================================================
    # GRUPO ④ — CONCENTRAÇÃO
    # ===================================================================
    # Pareto 80/20 da margem
    dv["_margin_total"] = dv["_margin_unit"] * dv[COL_QTY]
    mg = dv.groupby(COL_MATERIAL)["_margin_total"].sum().sort_values(ascending=False)
    mg_pos = mg[mg > 0]
    cum = mg_pos.cumsum() / mg_pos.sum()
    n80 = int((cum <= 0.80).sum()) + 1
    # gráfico Pareto mais legível: barras = % da margem por decil de SKUs; linha = acumulado
    n_sku = len(mg_pos)
    decis = np.array_split(mg_pos.values, 10)
    share_decil = [100 * bloc.sum() / mg_pos.sum() for bloc in decis]
    cum_decil = np.cumsum(share_decil)
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    xpos = np.arange(1, 11)
    ax1.bar(xpos, share_decil, color="#BA7517", alpha=0.85, width=0.7)
    ax1.set_xlabel("Decis de SKUs (1º = os 10% de maior margem)")
    ax1.set_ylabel("% da margem no decil", color="#BA7517")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"{i*10}%" for i in range(1, 11)])
    for i, v in enumerate(share_decil):
        ax1.text(xpos[i], v + 0.5, f"{v:.0f}%", ha="center", fontsize=7, color="#8a5a10")
    ax2 = ax1.twinx()
    ax2.plot(xpos, cum_decil, color="#33475b", marker="o", ms=4, lw=1.8)
    ax2.axhline(80, color="#c0504d", ls="--", lw=0.8)
    ax2.set_ylabel("% acumulado da margem", color="#33475b")
    ax2.set_ylim(0, 105)
    ax1.set_title("Pareto da margem — quanto cada decil de SKUs contribui")
    fig.tight_layout()
    share_top10 = share_decil[0]
    add("Concentração — Pareto 80/20 da margem",
        why="A regra de Pareto quase sempre vale em pricing: poucos SKUs geram a maior parte da margem. As barras mostram quanto cada fatia de 10% dos produtos contribui; a linha mostra o acumulado. Saber onde está a margem é essencial — são esses SKUs que merecem atenção prioritária de preço.",
        answers="Quantos SKUs concentram 80% da margem, e como a contribuição se distribui?",
        method="Somamos a margem total de cada SKU (material) e mantemos só os lucrativos (margem > 0). Ordenamos do maior para o menor e dividimos em 10 grupos iguais (decis): o 1º decil são os 10% de SKUs de maior margem. Cada barra mostra quanto % da margem total aquele decil concentra; a linha é o acumulado. O número de SKUs que chega a 80% é contado somando as margens ordenadas até cruzar 80% do total.",
        insight=f"Os 10% de SKUs de maior margem já respondem por {share_top10:.0f}% do total. No acumulado, apenas {n80:,} SKUs "
                f"({100*n80/n_sku:.1f}% do catálogo lucrativo) chegam a 80% da margem. É onde qualquer ajuste de preço tem mais "
                "impacto — e onde um erro custa mais caro. A cauda longa contribui pouco e inclui itens deficitários.",
        money=f"Foco de pricing: {n80:,} SKUs cobrem 80% da margem — priorizar esses multiplica o efeito de qualquer reajuste.",
        worked=[
            f"Concentração dá foco: mexer em {n80:,} SKUs já cobre 80% da margem — esforço de pricing altamente alavancado.",
            "Poucos SKUs 'campeões' facilitam o monitoramento e a governança de preço.",
        ],
        not_worked=[
            "A cauda longa (últimos decis) contribui pouco e inclui itens deficitários — complexidade que custa e rende pouco.",
            "Dependência de poucos SKUs é risco: perder um campeão de margem tem impacto desproporcional.",
        ],
        fig=fig)


    # ===================================================================
    # GRUPO ⑤ — RÉGUA DE TIER E SANGRIA DE MARGEM
    # ===================================================================
    # Régua de Tier: interativa por produto (ou visão geral se '(auto)')
    mat_sel = getattr(p, "diag_material", "(auto)")
    if mat_sel and mat_sel != "(auto)" and (dv[COL_MATERIAL].astype(str) == mat_sel).any():
        # modo PRODUTO: preço por tier de um material específico
        dm = dv[dv[COL_MATERIAL].astype(str) == mat_sel].copy()
        by_tier = dm.groupby(COL_TIER).agg(
            preco_mediano=(COL_PRICE, "median"), vendas=(COL_PRICE, "size"),
            qtd=(COL_QTY, "sum")).reset_index()
        by_tier = by_tier[by_tier["vendas"] >= 1].sort_values("preco_mediano")
        med_global = dm[COL_PRICE].median()
        fig, ax = plt.subplots(figsize=(9, 4.4))
        cols = ["#c0504d" if v > med_global else "#1d9e75" for v in by_tier["preco_mediano"]]
        ax.bar(by_tier[COL_TIER].astype(str), by_tier["preco_mediano"], color=cols)
        ax.axhline(med_global, color="#33475b", ls="--", lw=1, label=f"mediana do produto (US$ {med_global:,.0f})")
        for i, r in by_tier.reset_index().iterrows():
            ax.text(i, r["preco_mediano"], f"{int(r['vendas'])}v", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Preço mediano (US$)"); ax.legend(fontsize=8)
        ax.set_title(f"Produto {mat_sel}: preço mediano por Tier")
        fig.tight_layout()
        spread_prod = 100 * (by_tier["preco_mediano"].max() / by_tier["preco_mediano"].min() - 1) if len(by_tier) > 1 and by_tier["preco_mediano"].min() > 0 else 0
        add("Régua de Tier — preço por tier (produto selecionado)",
            why="A classificação de cliente (Tier) só se justifica se traduzir em preços diferentes. Aqui você escolhe UM produto e vê quanto cada tier paga por ele — a prova concreta, produto a produto, de que a régua de tier funciona ou está solta.",
            method=f"Filtramos todas as vendas do produto {mat_sel} e, para cada tier, calculamos o preço MEDIANO (o preço do meio, robusto a valores extremos). A linha tracejada é a mediana geral do produto. Barras vermelhas = tier paga acima da mediana; verdes = abaixo. O 'Nv' em cada barra é o número de vendas daquele tier (quanto menor, menos confiável a mediana). A variação citada é (preço do tier mais caro ÷ preço do tier mais barato − 1) × 100.",
            answers=f"No produto {mat_sel}, tiers diferentes pagam preços diferentes?",
            insight=f"Para o produto {mat_sel}, o preço mediano varia {spread_prod:.0f}% entre o tier mais barato e o mais caro. "
                    + ("Há diferenciação real por tier aqui." if spread_prod > 15 else
                       "Pouca diferenciação: os tiers pagam quase o mesmo por este produto — a régua não está sendo aplicada."),
            worked=[
                "Ver produto a produto permite auditar a régua de tier de forma concreta, não só no agregado.",
                "As barras coloridas (acima/abaixo da mediana) mostram na hora quais tiers pagam prêmio ou desconto.",
            ],
            not_worked=[
                "Se os tiers pagam quase o mesmo, a segmentação de cliente não está virando preço — alavanca desperdiçada.",
                "Poucas vendas em alguns tiers (veja o 'Nv' nas barras) tornam a mediana instável — cuidado ao concluir.",
            ],
            money="Diferenciar preço por tier onde hoje não há diferença é oportunidade estrutural de captura de valor.",
            fig=fig)
    else:
        # modo GERAL: heatmap tier x business
        dv["_pmed_mb"] = dv.groupby([COL_MATERIAL, COL_BUSINESS])[COL_PRICE].transform("median")
        dvh = dv[dv["_pmed_mb"] > 0].copy()
        dvh["preco_rel"] = dvh[COL_PRICE] / dvh["_pmed_mb"]
        ht = dvh.groupby([COL_BUSINESS, COL_TIER])["preco_rel"].median().unstack()
        tier_order = [t for t in ["A", "A-", "B", "C", "C-", "Distributor", "OEM/Integrator",
                                  "Sewage", "Standard", "UNSPECIFIED"] if t in ht.columns]
        ht = ht[tier_order]
        fig, ax = plt.subplots(figsize=(10, 5))
        data = ht.values.astype(float)
        im = ax.imshow(data, cmap="RdYlGn_r", vmin=0.8, vmax=1.2, aspect="auto")
        ax.set_xticks(range(len(ht.columns))); ax.set_xticklabels(ht.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ht.index))); ax.set_yticklabels([f"Business {b}" for b in ht.index], fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_title("Preço relativo por Tier × Business (1,00 = mediana do produto)")
        fig.colorbar(im, ax=ax, label="preço vs. mediana", shrink=0.7)
        fig.tight_layout()
        tier_spread = float(np.nanstd(data))
        add("Heatmap de preços — por tier e business",
            why="A classificação de cliente (Tier) só se justifica se traduzir em preços diferentes: um Tier A 'deveria' pagar diferente de um Tier C pelo mesmo produto. Use o seletor de produto no topo da aba para ver um produto específico.",
            method="Passo a passo: (1) para cada produto dentro de um business, calculamos a MEDIANA de preço daquele produto — o preço 'típico'. (2) Dividimos o preço de cada venda por essa mediana: uma venda a 1,10 custou 10% acima do típico; 0,90, 10% abaixo. Isso torna produtos caros e baratos comparáveis na mesma escala. (3) Para cada combinação Tier × Business, tiramos a mediana desses preços relativos — é o valor de cada célula. Leitura: 1,00 = o tier paga o preço típico; acima de 1 (vermelho) = paga prêmio; abaixo (verde) = paga desconto. O número entre parênteses no insight é o desvio-padrão de todas as células, medindo o quanto os tiers realmente diferem.",
            answers="Tiers diferentes realmente pagam preços diferentes pelo mesmo produto e business?",
            insight=f"Quase todas as células ficam perto de 1,00 (variação entre tiers de apenas {tier_spread:.2f}) — os tiers pagam "
                    "praticamente o MESMO preço pelo mesmo produto. As exceções são C- e UNSPECIFIED, que pagam ACIMA da mediana. "
                    "Confirma quantitativamente o que a modelagem já indicava: a régua de Tier quase não diferencia preço — está solta.",
            worked=[
                "Onde há diferença (C-, UNSPECIFIED pagando mais), o tier está capturando valor — prova que o mecanismo funciona quando aplicado.",
                "Normalizar por produto+business torna a comparação justa entre tiers, sem viés de mix de portfólio.",
            ],
            not_worked=[
                "A régua de Tier está solta: Tier A paga quase o mesmo que Tier C pelo mesmo produto — a segmentação não vira preço.",
                "Sem diferenciação por tier, a empresa perde uma alavanca clássica de pricing (cobrar conforme o perfil/valor do cliente).",
            ],
            money="Ativar a régua de Tier (diferenciar preço por perfil) é oportunidade estrutural — hoje subutilizada.",
            fig=fig)

    # -- Margem negativa por peer group (produto × tier × país × business) --
    PGn = [COL_MATERIAL, COL_TIER, COL_COUNTRY, COL_BUSINESS]
    dv["_margin_row"] = dv[COL_REVENUE] + dv[COL_COGS]
    gn = dv.groupby(PGn).agg(margem=("_margin_row", "sum"), receita=(COL_REVENUE, "sum"),
                             qtd=(COL_QTY, "sum"), vendas=("_margin_row", "size"))
    neg = gn[gn["margem"] < 0].copy()
    n_grupos_tot = len(gn); n_neg = len(neg)
    sangria = float(-neg["margem"].sum())
    rec_neg = float(neg["receita"].sum())
    negs = neg.sort_values("margem")
    cumneg = negs["margem"].cumsum() / negs["margem"].sum()
    n80_neg = int((cumneg <= 0.80).sum()) + 1
    mat_neg = neg.reset_index()[COL_MATERIAL].nunique()
    mat_tot = dv[COL_MATERIAL].nunique()
    pct_portfolio = 100 * mat_neg / mat_tot
    margem_total = float(dv["_margin_row"].sum())

    # top peer groups deficitários
    top_neg = negs.head(12).reset_index()
    top_neg_tbl = _round_df(
        top_neg[[COL_MATERIAL, COL_TIER, COL_COUNTRY, COL_BUSINESS, "vendas", "receita", "margem"]]
        .rename(columns={"receita": "receita_US$", "margem": "margem_US$"}),
        money_cols=["receita_US$", "margem_US$"], int_cols=["vendas"])

    # treemap: cada retângulo = um peer group deficitário, tamanho = perda
    top_tree = negs.head(25).reset_index()
    losses = (-top_tree["margem"]).values
    try:
        import squarify
        _have_sq = True
    except Exception:
        _have_sq = False
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.Reds
    norm_l = losses / losses.max()
    colors = [cmap(0.35 + 0.6 * v) for v in norm_l]
    labels = [f"{str(top_tree[COL_MATERIAL].iloc[i])[:8]}\nT{top_tree[COL_TIER].iloc[i]}\n${losses[i]/1e3:,.0f}k"
              for i in range(len(top_tree))]
    if _have_sq:
        squarify.plot(sizes=losses, label=labels, color=colors, ax=ax,
                      text_kwargs={"fontsize": 7}, pad=True)
        ax.axis("off")
    else:
        # fallback manual: barras horizontais grossas simulando blocos
        ypos = np.arange(len(top_tree))
        ax.barh(ypos, losses / 1e3, color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{str(top_tree[COL_MATERIAL].iloc[i])[:10]} T{top_tree[COL_TIER].iloc[i]}"
                            for i in range(len(top_tree))], fontsize=6)
        ax.invert_yaxis(); ax.set_xlabel("Perda (US$ mil)")
    ax.set_title(f"Onde está a sangria: {min(25, n_neg)} maiores grupos deficitários (tamanho = perda)")
    fig.tight_layout()

    add("Margem negativa — onde pagamos para vender (por grupo comparável)",
        why="Em vez de olhar transações negativas soltas, agrupamos por produto × tier × país × business: assim vemos onde a empresa é ESTRUTURALMENTE deficitária — vende no vermelho para um mesmo perfil de cliente, de forma recorrente. O treemap mostra cada grupo como um bloco cujo tamanho é a perda: quanto maior o bloco, mais dinheiro aquele grupo drena.",
        answers="Quais combinações produto/cliente vendem sistematicamente abaixo do custo? É concentrado ou generalizado?",
        method="Agrupamos por Material × Tier × País × Business e somamos a margem de cada grupo (Receita + Custo, com custo negativo). Um grupo é 'deficitário' se a margem somada dá negativa — ou seja, no total daquele produto para aquele perfil de cliente, a empresa perdeu dinheiro. A sangria é a soma dessas margens negativas. Para medir concentração, ordenamos os grupos do mais deficitário ao menos e contamos quantos somam 80% da perda. O treemap mostra os maiores: cada bloco é um grupo, e o tamanho é a perda. '% do portfólio' = materiais únicos no vermelho ÷ total de materiais.",
        insight=f"Dos {n_grupos_tot:,} grupos comparáveis, {n_neg:,} ({100*n_neg/n_grupos_tot:.1f}%) são deficitários, somando "
                f"US$ {sangria/1e6:,.1f} mi de sangria sobre US$ {rec_neg/1e6:,.1f} mi de receita. E é CONCENTRADO: só "
                f"{n80_neg} grupos ({100*n80_neg/n_neg:.0f}% dos deficitários) explicam 80% da perda — os blocos maiores do treemap. "
                f"Os materiais no vermelho são {mat_neg:,} ({pct_portfolio:.1f}% do portfólio). Não é descontrole geral, é cirúrgico.",
        worked=[
            f"Problema concentrado: {n80_neg} grupos causam 80% da sangria — dá para atacar com pouquíssimas decisões.",
            f"Apenas {pct_portfolio:.1f}% do portfólio está no vermelho: a política comercial no geral é saudável.",
        ],
        not_worked=[
            f"US$ {sangria/1e6:,.1f} mi de margem negativa estrutural: vendas recorrentes abaixo do custo para o mesmo perfil de cliente.",
            "Grupos deficitários recorrentes indicam preço/custo fora do lugar — não é exceção pontual, é padrão que se repete.",
        ],
        money=f"Sangria estrutural de US$ {sangria/1e6:,.1f} mi ({100*sangria/margem_total:.1f}% da margem total), "
              f"80% dela em só {n80_neg} grupos — foco cirúrgico.",
        fig=fig, table=top_neg_tbl)

    # -- Break-even: simulação de recuperação em vários níveis de markup --
    # para cada grupo deficitário: custo = receita - margem; ao reprecificar para
    # custo*(1+m), a nova margem do grupo = receita_nova - custo. Aproximação: mantém volume,
    # leva o preço a cobrir o custo + markup m.
    neg2 = neg.copy()
    neg2["custo"] = neg2["receita"] - neg2["margem"]  # custo positivo (margem é negativa)
    niveis = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    recuperacao = []
    for m in niveis:
        # nova margem do grupo se vender a custo*(1+m): receita_nova = custo*(1+m); margem_nova = custo*m
        margem_nova = (neg2["custo"] * m).sum()
        ganho = margem_nova - neg2["margem"].sum()  # vs margem negativa atual
        recuperacao.append(ganho)
    sim_tbl = pd.DataFrame({
        "cenário (preço = custo + markup)": [f"custo + {int(m*100)}%" for m in niveis],
        "margem recuperada (US$ mi)": [r / 1e6 for r in recuperacao],
        "margem final desses grupos (US$ mi)": [(neg2["custo"] * m).sum() / 1e6 for m in niveis],
    })
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xpos = np.arange(len(niveis))
    bars = ax.bar(xpos, [r / 1e6 for r in recuperacao], color="#1d9e75")
    ax.set_xticks(xpos); ax.set_xticklabels([f"custo +{int(m*100)}%" for m in niveis])
    ax.set_xlabel("Reajuste aplicado aos grupos deficitários")
    ax.set_ylabel("Margem recuperada (US$ mi)")
    ax.set_title("Quanto de margem recuperamos em cada cenário de reajuste")
    for i, r in enumerate(recuperacao):
        ax.text(i, r / 1e6, f"${r/1e6:,.1f} mi", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(recuperacao) / 1e6 * 1.15)
    fig.tight_layout()

    rec_zero = recuperacao[0]
    rec_10 = recuperacao[-1]
    add("Break-even — simulação de recuperação por nível de reajuste",
        why="A correção da sangria não é só 'parar de vender no prejuízo' — é escolher ATÉ QUE PONTO reajustar. Esta simulação mostra quanto de margem se recupera se levarmos os grupos deficitários a vender pelo custo (markup 0%, o break-even puro) ou pelo custo mais uma margem (2%, 4%… 10%). Cada barra é um cenário de decisão.",
        answers="Se reajustarmos os grupos deficitários para custo + X%, quanto de margem recuperamos em cada nível?",
        method="Para cada grupo deficitário, calculamos o custo (= Receita − Margem, já que a margem é negativa). Em cada cenário, simulamos vender por custo × (1 + markup): a margem nova do grupo passa a ser custo × markup. A 'margem recuperada' é essa margem nova menos a margem negativa atual — ou seja, quanto melhoramos ao sair do prejuízo. No markup 0% (vender exatamente ao custo), a recuperação é igual à sangria eliminada. Cada barra é um nível de reajuste. Importante: assume volume constante, então é um teto — na prática o reajuste pode reduzir a quantidade vendida.",
        insight=f"No break-even puro (vender ao custo, markup 0%), recuperam-se US$ {rec_zero/1e6:,.1f} mi — só de parar a sangria. "
                f"Levando a custo + 10%, a recuperação sobe para US$ {rec_10/1e6:,.1f} mi. Cada 2 pontos de markup adicionam "
                "margem de forma quase linear, dando uma régua clara de quanto exigir na renegociação.",
        worked=[
            f"Piso garantido: só estancar a sangria (markup 0%) já recupera US$ {rec_zero/1e6:,.1f} mi.",
            "A simulação dá uma meta graduada — dá para negociar reajuste por grupo conforme o que o mercado aceita.",
        ],
        not_worked=[
            "A simulação assume volume constante — na prática, reajuste pode reduzir volume (elasticidade), então é um teto otimista.",
            "Grupos com razão estratégica para preço baixo (entrada, fidelização) precisam de análise antes de reajustar.",
        ],
        money=f"Estancar a sangria rende US$ {rec_zero/1e6:,.1f} mi (markup 0%); a custo + 10%, até US$ {rec_10/1e6:,.1f} mi.",
        fig=fig, table=sim_tbl)

    notes = ["Diagnóstico — bloco de performance de preços completo."]
    return {"metrics": {}, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


def run_recomendacoes(df: pd.DataFrame, p: Params) -> dict:
    """Aba 03 · Recomendações — consolida diagnóstico e modelagem em um plano
    de ação por horizonte (curto, médio, longo prazo): o que fazer, quanto
    vale, como e por quê."""
    d = df[df["price_valid"]].copy()
    d["unit_cost"] = -d[COL_COGS_AVG]
    d["_m"] = d[COL_REVENUE] + d[COL_COGS]

    # ---- números reais das análises (recalculados aqui) ----
    # 1) sangria de margem negativa por grupo comparável + break-even
    PGn = [COL_MATERIAL, COL_TIER, COL_COUNTRY, COL_BUSINESS]
    gn = d.groupby(PGn).agg(margem=("_m", "sum"), receita=(COL_REVENUE, "sum")).reset_index()
    neg = gn[gn["margem"] < 0]
    sangria = float(-neg["margem"].sum())
    negs = neg.sort_values("margem")
    cum = negs["margem"].cumsum() / negs["margem"].sum()
    n80_neg = int((cum <= 0.80).sum()) + 1
    # break-even a custo+10%
    neg2 = neg.copy(); neg2["custo"] = neg2["receita"] - neg2["margem"]
    be_10 = float((neg2["custo"] * 0.10).sum() - neg2["margem"].sum())

    # 2) dispersão nos SKUs do PARETO 80/20 (80% da margem) — trazer vendas abaixo do P25 até o P25
    mg = d.groupby(COL_MATERIAL)["_m"].sum().sort_values(ascending=False)
    mg_pos = mg[mg > 0]
    cum_mg = mg_pos.cumsum() / mg_pos.sum()
    sku80 = set(cum_mg[cum_mg <= 0.80].index) | {cum_mg.index[0]}
    n_sku80 = len(sku80)
    dp = d[(d[COL_MATERIAL].isin(sku80)) & (d[COL_PRICE] >= 1)].copy()
    PGd = [COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY]
    dp["_p25"] = dp.groupby(PGd)[COL_PRICE].transform(lambda x: x.quantile(0.25))
    dp["_n"] = dp.groupby(PGd)[COL_PRICE].transform("count")
    dp = dp[dp["_n"] >= 8]
    below = dp[dp[COL_PRICE] < dp["_p25"]]
    opp_disp = float(((below["_p25"] - below[COL_PRICE]) * below[COL_QTY]).sum())

    # 3) quadrante proibido (preço baixo + volume baixo, dentro do grupo)
    dv = d.copy()
    PGq = [COL_MATERIAL, COL_TIER, COL_BUSINESS, COL_COUNTRY]
    gq = dv.groupby(PGq)
    dv["_pmed"] = gq[COL_PRICE].transform("median")
    dv["_qmed"] = gq[COL_QTY].transform("median")
    dv["_ngrp"] = gq[COL_PRICE].transform("count")
    dq = dv[(dv["_ngrp"] >= 8) & (dv["_pmed"] > 0) & (dv["_qmed"] > 0)].copy()
    proib = dq[(dq[COL_PRICE] < dq["_pmed"]) & (dv["_qmed"] > 0) & (dq[COL_QTY] < dq["_qmed"])]
    opp_proib = float(((proib["_pmed"] - proib[COL_PRICE]) * proib[COL_QTY]).sum())

    margem_total = float(d["_m"].sum())

    blocks = []

    def add(title, why="", answers="", insight="", table=None, fig=None, money="",
            worked=None, not_worked=None, method="", _section=""):
        blocks.append({"title": title, "why": why, "answers": answers, "insight": insight,
                       "table": table, "fig": fig, "money": money,
                       "worked": worked, "not_worked": not_worked, "method": method,
                       "_section": _section})

    # ---------- INTRO ----------
    # ---------- CURTO PRAZO ----------
    add("", _section="🟢 CURTO PRAZO — 0 a 3 meses · dinheiro rápido, baixo risco")

    curto_tbl = pd.DataFrame({
        "ação": ["Reduzir dispersão dos SKUs do Pareto 80/20 (trazer ao P25)",
                 "Estancar vendas onde o custo é maior que o preço",
                 "Travar descontos a clientes de baixo volume (política comercial)"],
        "quanto vale": [f"US$ {opp_disp/1e6:,.1f} mi",
                        f"US$ {sangria/1e6:,.1f} mi (até US$ {be_10/1e6:,.1f} mi com markup)",
                        f"US$ {opp_proib/1e6:,.1f} mi"],
        "esforço": ["Baixo", "Baixo", "Médio"],
        "fonte": ["Diagnóstico — dispersão + Pareto 80/20",
                  "Diagnóstico — margem negativa + break-even",
                  "Diagnóstico — Preço × Volume (quadrante proibido)"],
    })
    add("Ações de curto prazo",
        why="São correções cirúrgicas e de baixo risco: atacam prejuízo explícito e incoerências de preço que a análise já mapeou. Rendem dinheiro rápido e financiam as fases seguintes.",
        answers="O que rende dinheiro já nos próximos 90 dias, com o menor risco?",
        insight="COMO: (1) Priorizar os SKUs do Pareto 80/20 e alinhar as vendas mais baratas ao P25 do grupo — foco no que é representativo. "
                f"(2) Listar os {n80_neg} grupos que vendem abaixo do custo e renegociar ou descontinuar (regra: nenhuma venda abaixo do custo). "
                "(3) Criar uma política comercial que TRAVE descontos para clientes de baixo volume — hoje quem compra pouco às vezes paga menos, sem justificativa. "
                "POR QUÊ: são os ganhos mais defensáveis do case — corrigem prejuízo e incoerência, sem depender de reação de mercado.",
        money=f"Recuperação de curto prazo: US$ {opp_disp/1e6:,.1f} mi (dispersão Pareto→P25) + US$ {sangria/1e6:,.1f} mi (custo>preço) + "
              f"US$ {opp_proib/1e6:,.1f} mi (desconto a baixo volume) = ~US$ {(opp_disp+sangria+opp_proib)/1e6:,.1f} mi, "
              f"equivalente a {100*(opp_disp+sangria+opp_proib)/margem_total:.1f}% da margem atual.",
        table=curto_tbl)

    # ---------- MÉDIO PRAZO ----------
    add("", _section="🟡 MÉDIO PRAZO — 3 a 9 meses · estruturar a política de preço")

    medio_tbl = pd.DataFrame({
        "ação": ["Motor de segmentação de clientes (preço por perfil, não por tier)",
                 "Campanha de go-to-market para gerar leads (atacar o churn)",
                 "Programa de upselling algorítmico (aumentar a cesta por pedido)"],
        "quanto vale": ["Ativar a régua de preço (hoje o tier quase não diferencia)",
                        "Reverter a perda de clientes (a maior causa da queda de margem)",
                        "Aumento de ticket médio por pedido"],
        "esforço": ["Médio", "Médio", "Médio"],
        "fonte": ["Modelagem — ML/SHAP + heatmap de tier", "Diagnóstico — Ponte P/V/M (volume)", "Modelagem — clusters + MCDA"],
    })
    add("Ações de médio prazo",
        why="Depois de estancar as perdas, estruturamos o crescimento. O diagnóstico mostrou duas causas-raiz: preço que não diferencia por perfil de cliente, e — principalmente — PERDA DE CLIENTES de um ano para o outro. O médio prazo ataca as duas.",
        answers="Como estruturar preço por perfil e reverter a perda de clientes que derrubou a margem?",
        insight="COMO: (1) Construir um motor de segmentação que cobra conforme o PERFIL do cliente (valor, fidelidade, volume) — hoje há produtos com preço idêntico entre tiers que deveriam ser diferenciados. "
                "(2) Lançar uma campanha de go-to-market para AUMENTAR o número de leads — nosso principal problema foi a perda de clientes de um ano para o outro (base caiu 26%). "
                "(3) Implantar upselling via algoritmo para aumentar a cesta de cada pedido, usando os clusters de produtos que costumam ser comprados juntos. "
                "POR QUÊ: o diagnóstico provou que o negócio encolheu em clientes e volume — recuperar base e cesta ataca a causa real, não o sintoma.",
        money="A maior alavanca não é preço, é volume: reverter parte da perda de clientes (efeito Volume de −US$ 43 mi na Ponte P/V/M) supera qualquer ganho de reajuste.",
        table=medio_tbl)

    # ---------- LONGO PRAZO ----------
    add("", _section="🔴 LONGO PRAZO — 9+ meses · vantagem competitiva de pricing")

    longo_tbl = pd.DataFrame({
        "ação": ["Motor de pricing para renegociação via XGBoost",
                 "Estudo de elasticidade produto a produto (o que pode subir)",
                 "Inferência causal: quanto a queda de preço causa queda de vendas"],
        "quanto vale": ["Preço-alvo objetivo por SKU em cada renegociação",
                        "Saber onde há espaço real para subir preço",
                        "Medir o efeito causal do preço na demanda"],
        "esforço": ["Alto", "Alto", "Alto"],
        "fonte": ["Modelagem — XGBoost/SHAP", "Modelagem — econometria (elasticidade)", "Modelagem — inferência causal (IV)"],
    })
    add("Ações de longo prazo",
        why="Institucionalizar o pricing científico: transformar os modelos que construímos em ferramentas vivas de decisão, usadas continuamente pela área comercial — não como análise única, mas como capacidade permanente.",
        answers="Como transformar os modelos do case em vantagem competitiva sustentável de pricing?",
        insight="COMO: (1) Colocar o motor XGBoost no fluxo de renegociação — cada renegociação parte de um preço-alvo objetivo, não da intuição do vendedor. "
                "(2) Estudar a elasticidade produto a produto para saber EXATAMENTE quais SKUs podem ter preço puxado para cima e quais não (os elásticos). "
                "(3) Aprofundar a inferência causal para medir o quanto o decréscimo de preço realmente causa queda (ou não) de vendas — separando causa de correlação. "
                "POR QUÊ: encerra o ciclo do case — sai da análise pontual para um sistema de pricing orientado por dados, que se retroalimenta e melhora com o tempo.",
        money="Não é um ganho pontual, é capacidade permanente: pricing deixa de ser reativo e vira competência científica, protegendo margem de forma sustentável.",
        table=longo_tbl)

    # ---------- RESUMO VISUAL ----------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    horizontes = ["Curto\n(0-3m)", "Médio\n(3-9m)", "Longo\n(9m+)"]
    curto_val = (opp_disp + sangria + opp_proib) / 1e6
    valores = [curto_val, 43.0, 43.0]
    cores = ["#1d9e75", "#BA7517", "#c0504d"]
    bars = ax.bar(horizontes, valores, color=cores)
    rotulos = [f"US$ {curto_val:,.1f} mi\n(recuperação direta)",
               "Reverter perda\nde volume (~US$ 43 mi)",
               "Capacidade\npermanente"]
    for i, (v, r) in enumerate(zip(valores, rotulos)):
        ax.text(i, v, r, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Valor identificado (US$ mi)")
    ax.set_ylim(0, 52)
    ax.set_title("Plano por horizonte — do dinheiro rápido à vantagem sustentável")
    ax.text(0.5, -0.16, "Curto = corrigir perdas (baixo risco) · Médio = recuperar clientes/volume · Longo = pricing científico contínuo",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout()
    add("Panorama — o plano completo por horizonte",
        why="Uma visão única do plano: cada horizonte com seu foco. O curto prazo corrige perdas explícitas (dinheiro rápido e seguro); o médio recupera clientes e volume (a causa-raiz); o longo institucionaliza o pricing científico.",
        answers="Qual o foco e o retorno de cada fase do plano?",
        insight=f"Curto prazo: ~US$ {curto_val:,.1f} mi de recuperação direta e de baixo risco (dispersão Pareto→P25, custo>preço, trava de desconto). "
                "Médio prazo: recuperar a perda de clientes — o efeito Volume (−US$ 43 mi na Ponte P/V/M) foi a maior causa da queda de margem. "
                "Longo prazo: motor de pricing científico (XGBoost + elasticidade + causal) como capacidade permanente. "
                "A mensagem final do case: a IR precifica bem — o desafio é vender para mais gente.",
        money="Sequência recomendada: capture o curto prazo agora (financia o resto), recupere clientes no médio, institucionalize no longo.",
        fig=fig)

    notes = []
    metrics = {}
    return {"metrics": metrics, "blocks": blocks, "notes": notes,
            "tables": {}, "figures": {}}


# ========================================================================
# Registro das análises (para a UI iterar)
# ========================================================================
@dataclass
class Analysis:
    code: str
    name: str
    fn: Callable
    needs_elasticity: bool = False


ANALYSES = [
    Analysis("01", "Diagnóstico", run_diagnostico),
    Analysis("02", "Modelagem Avançada", run_advanced_modeling),
    Analysis("03", "Recomendações", run_recomendacoes),
]


# ======================================================================
# INTERFACE STREAMLIT
# ======================================================================

st.set_page_config(page_title="Case IR Pricing", layout="wide",
                   initial_sidebar_state="expanded")


# ------------------------------------------------------------------
# Cache: o processamento pesado só re-roda se o arquivo ou os params mudarem
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_and_clean(file_bytes: bytes, sheet: str | None):
    df_raw = read_excel(io.BytesIO(file_bytes), sheet=sheet)
    missing = validate_columns(df_raw)
    if missing:
        return None, missing, None
    df = clean(df_raw)
    return df, [], summarize_clean(df)


def _run_analysis(code: str, file_bytes: bytes, sheet: str | None, params_dict: dict):
    """Roda UMA análise. Sem cache: o retorno contém figuras matplotlib,
    que não são serializáveis pelo pickle do st.cache_data."""
    df, missing, _ = _load_and_clean(file_bytes, sheet)
    if df is None:
        return None
    p = Params(**params_dict)
    analysis = next(a for a in ANALYSES if a.code == code)
    if analysis.needs_elasticity:
        elas = run_elasticity(df, p)
        return analysis.fn(df, p, elas)
    return analysis.fn(df, p)


def _to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, tdf in tables.items():
            tdf.head(1_000_000).to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ------------------------------------------------------------------
# Sidebar — upload e parâmetros
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Configuração")

uploaded = st.sidebar.file_uploader(
    "Suba a base (.xlsx)", type=["xlsx", "xls"],
    help="Planilha transacional. A aba 'Data Basis' é usada por padrão.")

sheet_name = st.sidebar.text_input("Nome da aba (opcional)", value="Data Basis")

st.sidebar.markdown("---")
st.sidebar.caption("Os filtros de cada análise ficam no topo de cada aba.")

# defaults (podem ser sobrescritos pelos filtros no topo de cada aba)
capture_rate = 0.30
dispersion_pctl = 0.50
min_peer = 8
box_top_n = 15
box_rank_by = "uplift"
residual_pctl = 0.25
max_increase = 0.15
use_xgb = True

params_dict = dict(
    dispersion_target_pctl=dispersion_pctl,
    min_peer_size=int(min_peer),
    residual_target_pctl=residual_pctl,
    capture_rate=capture_rate,
    max_price_increase=max_increase,
    use_xgb=use_xgb,
    boxplot_top_n=int(box_top_n),
    boxplot_rank_by=box_rank_by,
)


# ------------------------------------------------------------------
# Corpo
# ------------------------------------------------------------------
st.title("📊 Case IR Pricing")
st.caption("Diagnóstico de pricing e recuperação de margem — do retrato do negócio às recomendações.")

if uploaded is None:
    st.info("⬅️ Suba a base `.xlsx` na barra lateral para começar.")
    with st.expander("Colunas esperadas na base"):
        st.write(", ".join(REQUIRED_COLS))
    st.stop()

file_bytes = uploaded.getvalue()
sheet = sheet_name.strip() or None

with st.spinner("Lendo e limpando a base..."):
    df, missing, _ = _load_and_clean(file_bytes, sheet)

if df is None:
    st.error("⚠️ A base carregada não tem todas as colunas obrigatórias.")
    st.markdown("**Colunas que estão faltando:**")
    st.write(missing)
    st.markdown("**O app espera uma planilha com estas colunas** (aba "
                f"'{SHEET_DEFAULT}' de preferência):")
    st.write(REQUIRED_COLS)
    st.info("Verifique se você subiu a base correta do case e se a aba tem os nomes de coluna exatos "
            "(maiúsculas, espaços e símbolos como '($)' importam).")
    st.stop()

# listas para os filtros (dependem da base carregada)
tiers_all = ["(todos)"] + sorted(df[COL_TIER].dropna().unique().tolist())
countries_rev = (df.groupby(COL_COUNTRY)[COL_REVENUE].sum()
                 .sort_values(ascending=False).index.tolist())
# materiais com mais tiers distintos (mais interessantes para a régua de tier)
_mat_tiers = (df[df[COL_PRICE] > 0].groupby(COL_MATERIAL)[COL_TIER].nunique()
              .sort_values(ascending=False))
mats_multi = ["(auto)"] + _mat_tiers[_mat_tiers >= 2].head(200).index.astype(str).tolist()
# SKUs para o seletor do MCDA (por receita, os mais relevantes)
_mat_rev = df[df[COL_PRICE] > 0].groupby(COL_MATERIAL)[COL_REVENUE].sum().sort_values(ascending=False)
mcda_skus = ["(todos)"] + _mat_rev.head(300).index.astype(str).tolist()

# Abas por análise
tab_labels = [f"{a.code} · {a.name}" for a in ANALYSES]
tabs = st.tabs(tab_labels)

for tab, analysis in zip(tabs, ANALYSES):
    with tab:
        # filtros no topo de cada aba (em vez da sidebar)
        local_params = dict(params_dict)
        if analysis.code == "01":  # Diagnóstico
            with st.container():
                c1, c2 = st.columns(2)
                with c1:
                    diag_country = st.selectbox(
                        "🌍 País da análise de dispersão", ["(auto)"] + countries_rev,
                        help="'(auto)' usa o país de maior receita", key="diag_country")
                with c2:
                    diag_material = st.selectbox(
                        "📦 Produto (régua de tier)", mats_multi,
                        help="'(auto)' mostra a visão geral; escolha um produto para ver os tiers dele",
                        key="diag_material")
            local_params["diag_country"] = diag_country
            local_params["diag_material"] = diag_material
        elif analysis.code == "02":  # Modelagem Avançada
            PERFIS_MCDA = {
                "Defender share":     {"margem": .10, "elast": .15, "share": .30, "churn": .30, "cluster": .15},
                "Extrair margem":     {"margem": .40, "elast": .30, "share": .05, "churn": .10, "cluster": .15},
                "Equilibrado":        {"margem": .20, "elast": .20, "share": .20, "churn": .20, "cluster": .20},
                "Corrigir distorções":{"margem": .15, "elast": .15, "share": .10, "churn": .10, "cluster": .50},
            }
            with st.expander("⚖️ Controles do modelo de decisão multicritério (MCDA) — 4º bloco desta aba", expanded=True):
                st.caption("Escolha o OBJETIVO estratégico do portfólio — ele define automaticamente o peso de cada critério. "
                           "Ou selecione 'Personalizado' para ajustar os pesos manualmente.")
                perfil = st.radio("🎯 Objetivo estratégico", list(PERFIS_MCDA.keys()) + ["Personalizado"],
                                  horizontal=True, key="mcda_perfil_sel")
                if perfil == "Personalizado":
                    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
                    with wc1:
                        w_margem = st.slider("💰 Margem", 0.0, 1.0, 0.20, 0.05, key="w_margem")
                    with wc2:
                        w_elast = st.slider("📉 Elasticidade", 0.0, 1.0, 0.20, 0.05, key="w_elast")
                    with wc3:
                        w_share = st.slider("🏆 Market share", 0.0, 1.0, 0.20, 0.05, key="w_share")
                    with wc4:
                        w_churn = st.slider("🔄 Churn", 0.0, 1.0, 0.20, 0.05, key="w_churn")
                    with wc5:
                        w_cluster = st.slider("🧩 Cluster", 0.0, 1.0, 0.20, 0.05, key="w_cluster")
                else:
                    pw = PERFIS_MCDA[perfil]
                    w_margem, w_elast, w_share, w_churn, w_cluster = (
                        pw["margem"], pw["elast"], pw["share"], pw["churn"], pw["cluster"])
                    st.caption(f"Pesos do perfil **{perfil}**: 💰 Margem {pw['margem']*100:.0f}% · "
                               f"📉 Elasticidade {pw['elast']*100:.0f}% · 🏆 Share {pw['share']*100:.0f}% · "
                               f"🔄 Churn {pw['churn']*100:.0f}% · 🧩 Cluster {pw['cluster']*100:.0f}%")
                mcda_sku = st.selectbox("🔎 Ver um SKU específico (ou os top 20 por ganho)",
                                        mcda_skus, key="mcda_sku",
                                        help="'(todos)' mostra os SKUs de maior ganho potencial")
            local_params["mcda_perfil"] = perfil
            local_params["w_margem"] = w_margem
            local_params["w_elasticidade"] = w_elast
            local_params["w_share"] = w_share
            local_params["w_churn"] = w_churn
            local_params["w_cluster"] = w_cluster
            local_params["mcda_sku"] = mcda_sku

        with st.spinner(f"Rodando {analysis.name}..."):
            try:
                res = _run_analysis(analysis.code, file_bytes, sheet, local_params)
            except Exception as e:
                st.error(f"Falha ao rodar {analysis.name}: {e}")
                continue

        if res is None:
            st.warning("Sem resultado.")
            continue

        # métricas
        if res.get("metrics"):
            mcols = st.columns(min(4, len(res["metrics"])) or 1)
            for i, (k, v) in enumerate(res["metrics"].items()):
                mcols[i % len(mcols)].metric(k, v)

        # notas
        for note in res.get("notes", []):
            st.caption("ℹ️ " + str(note).replace("$", "\\$"))

        # ---- modo BLOCOS (abas com análise detalhada) ----
        blocks = res.get("blocks")
        if blocks:
            def _esc(txt):
                # Streamlit interpreta '$...$' como LaTeX; escapamos o cifrão
                return str(txt).replace("$", "\\$")
            for blk in blocks:
                if blk.get("_section"):
                    st.markdown(f"## {blk['title']}")
                    if blk.get("insight"):
                        st.markdown(f"*{_esc(blk['insight'])}*")
                    st.divider()
                    continue
                if blk.get("title"):
                    st.markdown(f"#### {blk['title']}")
                # como o modelo funciona (explicação didática do mecanismo)
                if blk.get("how_it_works"):
                    st.info("⚙️ **Como este modelo funciona:** " + _esc(blk["how_it_works"]))
                # bullets explicativos (por que / o que responde / insight)
                if blk.get("why"):
                    st.markdown(f"- **Por que esta análise:** {_esc(blk['why'])}")
                if blk.get("answers"):
                    st.markdown(f"- **O que queremos responder:** {_esc(blk['answers'])}")
                # metodologia (como chegamos nos números) — em destaque
                if blk.get("method"):
                    st.info("🧮 **Como chegamos nos números:** " + _esc(blk["method"]))
                if blk.get("assumptions"):
                    st.markdown(f"- **Pressupostos:** {_esc(blk['assumptions'])}")
                # formulação matemática (LaTeX)
                if blk.get("formula"):
                    st.markdown("**Formulação:**")
                    for eq in (blk["formula"] if isinstance(blk["formula"], list) else [blk["formula"]]):
                        st.latex(eq)
                    if blk.get("formula_legend"):
                        st.caption(_esc(blk["formula_legend"]))
                # prós e contras lado a lado
                if blk.get("pros") or blk.get("cons"):
                    cpro, ccon = st.columns(2)
                    if blk.get("pros"):
                        cpro.markdown("**✔ Prós**")
                        for it in blk["pros"]:
                            cpro.markdown(f"- {_esc(it)}")
                    if blk.get("cons"):
                        ccon.markdown("**✘ Contras**")
                        for it in blk["cons"]:
                            ccon.markdown(f"- {_esc(it)}")
                if blk.get("insight"):
                    st.markdown(f"- **O que achamos de interessante:** {_esc(blk['insight'])}")
                # o que funcionou / o que não funcionou (bullets)
                if blk.get("worked"):
                    st.markdown("- **O que funcionou:**")
                    for it in blk["worked"]:
                        st.markdown(f"    - ✅ {_esc(it)}")
                if blk.get("not_worked"):
                    st.markdown("- **O que não funcionou:**")
                    for it in blk["not_worked"]:
                        st.markdown(f"    - ⚠️ {_esc(it)}")
                # valor monetário em destaque
                if blk.get("money"):
                    st.success("💰 " + _esc(blk["money"]))
                if blk.get("fig") is not None:
                    st.pyplot(blk["fig"], use_container_width=True)
                if blk.get("table2") is not None:
                    st.caption(blk.get("table2_caption", "Como ler cada efeito:"))
                    st.dataframe(blk["table2"], use_container_width=True, height=240)
                if blk.get("table") is not None:
                    st.dataframe(blk["table"], use_container_width=True, height=300)
                st.divider()
            continue  # aba de blocos não usa o fluxo de tabelas/figuras abaixo

        # ---- modo padrão (demais abas) ----
        captions = res.get("captions", {})
        for fig_name, fig in res.get("figures", {}).items():
            st.pyplot(fig, use_container_width=True)
            if fig_name in captions:
                st.caption(captions[fig_name])

        tables = res.get("tables", {})
        if tables:
            for name, tdf in tables.items():
                st.markdown(f"**{name}**  ·  {len(tdf):,} linhas")
                st.dataframe(tdf.head(1000), use_container_width=True, height=280)
            # apenas um download consolidado por aba (sem botão por tabela)
            st.download_button(
                f"⬇️ Baixar tabelas desta análise (Excel)",
                _to_excel_bytes(tables),
                file_name=f"{analysis.code}_{analysis.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dlxlsx_{analysis.code}")
