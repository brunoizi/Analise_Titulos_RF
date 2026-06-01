"""
Engine do Scanner Quantitativo do Tesouro Direto — v2.

Novidades desta versão:
- compute_risk_metrics: Duration, Modified Duration, Convexidade, DV01.
- build_yield_curve: curva de juros por família (pré/IPCA+) a partir do TD.
- implied_inflation: inflação implícita via pares Pré vs IPCA+.
- fetch_anbima_ettj: busca ETTJ ANBIMA (opcional; fallback silencioso).
- simulate_carry_ipca: agora aceita aportes periódicos (DCA).
- simulate_sell_before_maturity: compara carrego vs venda antecipada (MtM).
- compute_fat_tail_index: índice MAD/STD (Taleb cap 4.4.1) para detectar
  caudas gordas nos retornos do PU, com backtest de estratégia de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import timedelta

URL_CSV = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/"
    "PrecoTaxaTesouroDireto.csv"
)


# ===================== Load =====================
def load_csv(url: str = URL_CSV, timeout: int = 60) -> pd.DataFrame:
    """Baixa o CSV oficial do Tesouro Transparente."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), sep=";", decimal=",", encoding="utf-8")
    df.columns = df.columns.str.strip()
    return df


# ===================== Prepare =====================
def prepare(df_raw: pd.DataFrame, taxa_col: str, pu_col: str | None = None) -> pd.DataFrame:
    """Renomeia/filtra/tipa as colunas necessárias."""
    rename = {
        "Tipo Titulo": "titulo",
        "Data Vencimento": "vencimento",
        "Data Base": "data",
        taxa_col: "taxa",
    }
    if pu_col:
        rename[pu_col] = "pu"

    needed = ["Tipo Titulo", "Data Vencimento", "Data Base", taxa_col]
    if pu_col:
        needed.append(pu_col)

    missing = [c for c in needed if c not in df_raw.columns]
    if missing:
        raise KeyError(
            f"Colunas ausentes: {missing}. Disponíveis: {list(df_raw.columns)}"
        )

    df = df_raw[needed].copy().rename(columns=rename)

    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df["vencimento"] = pd.to_datetime(df["vencimento"], dayfirst=True, errors="coerce")
    df["taxa"] = pd.to_numeric(df["taxa"], errors="coerce")
    if pu_col:
        df["pu"] = pd.to_numeric(df["pu"], errors="coerce")

    df = df.dropna(subset=["titulo", "vencimento", "data", "taxa"])
    if pu_col:
        df = df.dropna(subset=["pu"])

    return df.sort_values(["titulo", "vencimento", "data"]).reset_index(drop=True)


# ===================== Scanner (quartis + z) =====================
def quartil_label(x: float, q1: float, q2: float, q3: float) -> str:
    """J1 = taxa baixa; J4 = taxa alta (melhor p/ compra em reversão)."""
    if x <= q1:
        return "J1"
    if x <= q2:
        return "J2"
    if x <= q3:
        return "J3"
    return "J4"


def compute_signals_all(
    df: pd.DataFrame,
    windows_days=(180, 365, 730),
    min_points: int = 80,
) -> pd.DataFrame:
    """Para cada (título, vencimento) calcula J/Z em múltiplas janelas + score."""
    last_date = df["data"].max()
    df_last = df[df["data"] == last_date]

    rows = []
    for (titulo, vencimento), g in df.groupby(["titulo", "vencimento"]):
        cur = df_last[
            (df_last["titulo"] == titulo) & (df_last["vencimento"] == vencimento)
        ]
        if cur.empty:
            continue

        taxa_atual = float(cur["taxa"].iloc[-1])
        row = {
            "titulo": titulo,
            "vencimento": vencimento.date(),
            "taxa_atual": taxa_atual,
        }

        for wd in windows_days:
            start = last_date - timedelta(days=int(wd))
            s = g[(g["data"] >= start) & (g["data"] <= last_date)]["taxa"].dropna()
            if len(s) < min_points:
                row[f"J{wd}"] = np.nan
                row[f"Z{wd}"] = np.nan
                continue

            q1, q2, q3 = s.quantile([0.25, 0.50, 0.75]).values
            mu = float(s.mean())
            sd = float(s.std(ddof=0)) if float(s.std(ddof=0)) != 0 else np.nan

            row[f"J{wd}"] = quartil_label(taxa_atual, q1, q2, q3)
            row[f"Z{wd}"] = (
                (taxa_atual - mu) / sd if (sd and not np.isnan(sd)) else np.nan
            )

        def j_score(j):
            return {"J1": 0, "J2": 1, "J3": 2, "J4": 3}.get(j, 0)

        pts = (
            2 * j_score(row.get("J730"))
            + 1 * j_score(row.get("J365"))
            + 1.5 * (row.get("Z730") if pd.notna(row.get("Z730")) else 0)
            + 0.8 * (row.get("Z365") if pd.notna(row.get("Z365")) else 0)
        )
        concord = (
            row.get("J365") == row.get("J730")
            if (
                row.get("J365") in ["J1", "J2", "J3", "J4"]
                and row.get("J730") in ["J1", "J2", "J3", "J4"]
            )
            else False
        )
        row["concord_365_730"] = concord
        row["score"] = pts + (0.5 if concord else 0)

        rows.append(row)

    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


# ===================== Série de 1 título =====================
def get_series(
    df: pd.DataFrame, titulo: str, vencimento: pd.Timestamp
) -> pd.DataFrame:
    d = df[(df["titulo"] == titulo) & (df["vencimento"] == vencimento)].copy()
    return d.sort_values("data")


# ===================== Backtest =====================
def build_signal_series(
    d: pd.DataFrame, window_days: int = 365, min_points: int = 80
) -> pd.DataFrame:
    """Calcula J e Z ao longo do tempo com janela móvel."""
    d = d.sort_values("data").copy()
    j_list, z_list = [], []
    for i in range(len(d)):
        end = d.iloc[i]["data"]
        start = end - timedelta(days=int(window_days))
        s = d[(d["data"] >= start) & (d["data"] <= end)]["taxa"].dropna()
        if len(s) < min_points:
            j_list.append(np.nan)
            z_list.append(np.nan)
            continue
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75]).values
        mu = float(s.mean())
        sd = float(s.std(ddof=0)) if float(s.std(ddof=0)) != 0 else np.nan
        x = float(d.iloc[i]["taxa"])
        j_list.append(quartil_label(x, q1, q2, q3))
        z_list.append((x - mu) / sd if (sd and not np.isnan(sd)) else np.nan)

    out = d.copy()
    out[f"J{window_days}"] = j_list
    out[f"Z{window_days}"] = z_list
    return out


def forward_change(
    df_series: pd.DataFrame, col: str, horizons=(30, 60, 90, 180)
) -> pd.DataFrame:
    """Pega valor futuro pela primeira data >= data_atual + h."""
    df_series = df_series.sort_values("data").reset_index(drop=True)
    out = pd.DataFrame({"data": df_series["data"], col: df_series[col]})
    for h in horizons:
        target_dates = df_series["data"] + pd.to_timedelta(h, unit="D")
        future_vals = []
        for td in target_dates:
            j = df_series["data"].searchsorted(td, side="left")
            future_vals.append(df_series.iloc[j][col] if j < len(df_series) else np.nan)
        out[f"{col}_fwd_{h}d"] = future_vals
        out[f"delta_{col}_{h}d"] = out[f"{col}_fwd_{h}d"] - out[col]
        if col == "pu":
            out[f"ret_pu_{h}d"] = (out[f"{col}_fwd_{h}d"] / out[col]) - 1.0
    return out


def backtest_mean_reversion(
    df_full: pd.DataFrame,
    titulo: str,
    vencimento: pd.Timestamp,
    window_days: int = 365,
    min_points: int = 80,
    j_trigger: str = "J4",
    z_min: float = 1.0,
    horizons=(30, 60, 90, 180),
) -> dict:
    """Dispara sinal (J==j_trigger AND Z>=z_min) e mede taxa/PU à frente."""
    d = get_series(df_full, titulo, vencimento)
    d_sig = build_signal_series(d, window_days=window_days, min_points=min_points)

    if "pu" in d_sig.columns:
        fwd = forward_change(
            d_sig[
                ["data", "taxa", "pu", f"J{window_days}", f"Z{window_days}"]
            ].copy(),
            "pu",
            horizons=horizons,
        )
    else:
        fwd = None

    fwd_taxa = forward_change(
        d_sig[["data", "taxa", f"J{window_days}", f"Z{window_days}"]].copy(),
        "taxa",
        horizons=horizons,
    )

    mask = (d_sig[f"J{window_days}"] == j_trigger) & (
        d_sig[f"Z{window_days}"] >= z_min
    )

    events = d_sig.loc[
        mask, ["data", "taxa", f"J{window_days}", f"Z{window_days}"]
    ].copy()
    if "pu" in d_sig.columns:
        events = events.merge(d_sig[["data", "pu"]], on="data", how="left")

    events = events.merge(fwd_taxa.drop(columns=["taxa"]), on="data", how="left")
    if fwd is not None:
        events = events.merge(fwd.drop(columns=["pu"]), on="data", how="left")

    summary = {
        "n_events": int(len(events)),
        "window_days": window_days,
        "j_trigger": j_trigger,
        "z_min": z_min,
    }

    for h in horizons:
        dt = events.get(f"delta_taxa_{h}d")
        if dt is not None:
            dt_clean = dt.dropna()
            if len(dt_clean) > 0:
                summary[f"mean_delta_taxa_{h}d"] = float(np.nanmean(dt))
                summary[f"hit_taxa_down_{h}d"] = float(
                    np.nanmean((dt < 0).astype(float))
                )

        rp = events.get(f"ret_pu_{h}d")
        if rp is not None:
            rp_clean = rp.dropna()
            if len(rp_clean) > 0:
                summary[f"mean_ret_pu_{h}d"] = float(np.nanmean(rp))
                summary[f"median_ret_pu_{h}d"] = float(np.nanmedian(rp))
                summary[f"p25_ret_pu_{h}d"] = float(np.nanpercentile(rp_clean, 25))
                summary[f"p75_ret_pu_{h}d"] = float(np.nanpercentile(rp_clean, 75))
                summary[f"hit_pu_up_{h}d"] = float(
                    np.nanmean((rp > 0).astype(float))
                )
                std_rp = float(np.nanstd(rp))
                summary[f"sharpe_like_pu_{h}d"] = (
                    float(np.nanmean(rp) / std_rp) if std_rp > 0 else np.nan
                )

    return {"events": events, "summary": summary, "series": d_sig}


# ===================== Cenário PU~taxa (regressão) =====================
def estimate_pu_from_taxa(
    series: pd.DataFrame, taxa_future: float, lookback_days: int = 730
) -> dict:
    """Ajusta PU = a + b*taxa por regressão linear na janela recente."""
    s = series.sort_values("data").copy()
    end = s["data"].max()
    start = end - timedelta(days=int(lookback_days))
    w = s[(s["data"] >= start) & (s["data"] <= end)].dropna(subset=["taxa", "pu"])

    if len(w) < 30:
        raise ValueError("Poucos dados para regressão PU~taxa nessa janela.")

    x = w["taxa"].values
    y = w["pu"].values
    b, a = np.polyfit(x, y, 1)
    pu_est = a + b * taxa_future

    return {
        "a": float(a),
        "b": float(b),
        "pu_est": float(pu_est),
        "n": int(len(w)),
        "lookback_days": lookback_days,
    }


# ===================== Métricas de risco (duration, convexidade, DV01) =====================
def classify_family(titulo: str) -> str:
    """Classifica o título pela família (prefixada, IPCA+, Selic)."""
    t = titulo.lower()
    if "selic" in t:
        return "selic"
    if "ipca" in t or "educa" in t or "renda+" in t:
        return "ipca"
    if "pref" in t:
        return "pre"
    if "igpm" in t or "igp-m" in t:
        return "igpm"
    return "outro"


def _zero_coupon_metrics(pu: float, taxa_a: float, years_to_mat: float) -> dict:
    """Métricas fechadas para zero-cupom (LTN, NTN-B Principal, Selic)."""
    r = taxa_a
    T = max(years_to_mat, 1e-6)

    # Para zero-cupom: Duration = prazo; Modified Duration = T/(1+r)
    duration = T
    mod_duration = T / (1.0 + r)
    # Convexidade zero-cupom: T*(T+1)/(1+r)^2
    convexity = T * (T + 1.0) / (1.0 + r) ** 2
    # DV01: variação do PU para +1 bp na taxa
    dv01 = pu * mod_duration * 0.0001

    return {
        "duration_macaulay_a": duration,
        "duration_modified_a": mod_duration,
        "convexidade_a": convexity,
        "dv01_R$": dv01,
        "tipo_fluxo": "zero_cupom",
    }


def _coupon_bond_metrics(
    pu: float, taxa_a: float, years_to_mat: float, coupon_a: float = 0.10
) -> dict:
    """
    Métricas para título com cupom semestral (NTN-B padrão e NTN-F).
    Usa cupom de 10%a.a. por padrão (Tesouro IPCA+ com Juros Semestrais, NTN-F).
    Tesouro Prefixado com Juros Semestrais também paga 10% a.a.

    Aproxima o fluxo usando:
      - cupons semestrais no valor C = (1 + coupon_a)^0.5 - 1 sobre face (=1000 p/ NTN-B em termos reais)
      - principal no vencimento
    Como é aproximação, pode ter pequena diferença vs cálculo oficial ANBIMA,
    mas captura muito bem a sensibilidade (duration/convexidade).
    """
    r = taxa_a
    T = max(years_to_mat, 1e-6)

    # Cupom semestral
    c_semestre = (1 + coupon_a) ** 0.5 - 1.0

    # Fluxo: cupons em 0.5, 1.0, 1.5, ... até T (inclusive)
    # (aproximação: assume primeiro cupom em ~0.5a, último junto com principal em T)
    coupon_times = []
    t_next = 0.5
    while t_next < T:
        coupon_times.append(t_next)
        t_next += 0.5
    # vencimento final
    times = np.array(coupon_times + [T])
    # fluxos (em base 1, relativo ao face): C, C, ..., C + 1
    flows = np.full(len(times), c_semestre)
    flows[-1] = flows[-1] + 1.0

    # Desconta pelos juros YTM (tax oficial usa dias úteis / 252, aproximamos com anos)
    disc = (1.0 + r) ** (-times)
    pv = flows * disc
    pv_total = pv.sum()  # base 1 (face)

    # Macaulay duration (ponderada pelo PV)
    weights = pv / pv_total
    duration = float((weights * times).sum())
    mod_duration = duration / (1.0 + r)
    # Convexidade: soma de t*(t+1)/(1+r)^2 * pv / pv_total
    convexity = float(
        ((times * (times + 1.0)) * disc * flows).sum() / pv_total / (1.0 + r) ** 2
    )
    # DV01 sobre o PU informado (não sobre o modelo, pra refletir a posição real)
    dv01 = pu * mod_duration * 0.0001

    return {
        "duration_macaulay_a": duration,
        "duration_modified_a": mod_duration,
        "convexidade_a": convexity,
        "dv01_R$": dv01,
        "tipo_fluxo": "cupom_semestral",
    }


def compute_risk_metrics(
    titulo: str,
    taxa_a: float,
    pu: float,
    years_to_mat: float,
    coupon_a: float = 0.10,
) -> dict:
    """
    Calcula duration, modified duration, convexidade e DV01.

    Parâmetros:
    - taxa_a: taxa anualizada em decimal (ex.: 0.0793 para 7.93% a.a.).
      Se vier > 1, assume que está em % e divide por 100.
    - pu: PU em R$.
    - years_to_mat: anos até o vencimento (pode ser fracionário).
    - coupon_a: cupom anual nominal (default 10%, que é o padrão de NTN-B e NTN-F).

    Heurística de fluxo:
    - "Juros Semestrais" no nome → cupom
    - "Principal" no nome → zero-cupom
    - "Prefixado" simples → zero-cupom (LTN)
    - "IPCA+" sem Juros Semestrais → zero-cupom (NTN-B Principal)
    - "Selic" → zero-cupom (LFT, aproximação)
    """
    r = taxa_a / 100.0 if taxa_a > 1 else taxa_a
    t_low = titulo.lower()
    has_coupon = ("juros semestrais" in t_low) or (t_low.strip().endswith("ntn-f"))

    if has_coupon:
        m = _coupon_bond_metrics(pu, r, years_to_mat, coupon_a=coupon_a)
    else:
        m = _zero_coupon_metrics(pu, r, years_to_mat)

    m["titulo"] = titulo
    m["familia"] = classify_family(titulo)
    m["taxa_a"] = r
    m["pu"] = pu
    m["anos_ate_venc"] = years_to_mat
    return m


def pu_sensitivity_table(
    pu: float,
    taxa_a: float,
    years_to_mat: float,
    titulo: str,
    delta_bps_list=(-200, -100, -50, -25, 0, 25, 50, 100, 200),
    coupon_a: float = 0.10,
) -> pd.DataFrame:
    """
    Tabela de sensibilidade: qual o PU estimado se a taxa variar X bps.
    Usa aproximação de 2ª ordem:
        ΔPU/PU ≈ -ModDur * Δy + 0.5 * Conv * (Δy)^2
    """
    m = compute_risk_metrics(titulo, taxa_a, pu, years_to_mat, coupon_a=coupon_a)
    md = m["duration_modified_a"]
    cx = m["convexidade_a"]

    rows = []
    for bps in delta_bps_list:
        dy = bps / 10000.0  # bps para decimal
        ret = -md * dy + 0.5 * cx * dy**2
        pu_new = pu * (1 + ret)
        rows.append(
            {
                "delta_bps": bps,
                "delta_taxa_pp": bps / 100.0,  # em pontos percentuais
                "pu_estimado": pu_new,
                "variacao_R$": pu_new - pu,
                "variacao_%": ret * 100.0,
            }
        )
    return pd.DataFrame(rows)


# ===================== Curva de juros (TD) =====================
def build_yield_curve(
    df: pd.DataFrame,
    familia: str = "ipca",
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Monta a curva de juros do próprio Tesouro Direto para uma família.

    familia: "ipca", "pre", "selic"
    ref_date: data de referência (default = última data disponível)

    Retorna DataFrame com: titulo, vencimento, anos_ate_venc, taxa_atual
    ordenado por prazo.
    """
    if ref_date is None:
        ref_date = df["data"].max()
    else:
        ref_date = pd.Timestamp(ref_date).normalize()

    snap = df[df["data"] == ref_date].copy()
    if snap.empty:
        # procura a data mais próxima se a solicitada não existir
        ref_date = df["data"].max()
        snap = df[df["data"] == ref_date].copy()

    snap["familia"] = snap["titulo"].apply(classify_family)
    snap = snap[snap["familia"] == familia].copy()

    snap["anos_ate_venc"] = (snap["vencimento"] - ref_date).dt.days / 365.25
    snap = snap[snap["anos_ate_venc"] > 0]

    # para cada vencimento, pega a MAIOR taxa (geralmente só tem uma por dia mesmo)
    snap = (
        snap.sort_values(["vencimento", "taxa"], ascending=[True, False])
        .drop_duplicates(subset=["titulo", "vencimento"], keep="first")
        .sort_values("anos_ate_venc")
        .reset_index(drop=True)
    )

    return snap[["titulo", "vencimento", "anos_ate_venc", "taxa"]].rename(
        columns={"taxa": "taxa_atual"}
    )


# ===================== Inflação implícita =====================
def implied_inflation(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    tol_years: float = 0.75,
) -> pd.DataFrame:
    """
    Inflação implícita = [(1+pré)/(1+real) - 1], pareando pré e IPCA+ com
    vencimentos próximos (até tol_years de diferença).

    Retorna tabela de pares (pré, IPCA+) e a inflação implícita anualizada.
    """
    curva_pre = build_yield_curve(df, familia="pre", ref_date=ref_date)
    curva_ipca = build_yield_curve(df, familia="ipca", ref_date=ref_date)

    if curva_pre.empty or curva_ipca.empty:
        return pd.DataFrame()

    pairs = []
    for _, p in curva_pre.iterrows():
        # acha o IPCA+ mais próximo em prazo
        diffs = (curva_ipca["anos_ate_venc"] - p["anos_ate_venc"]).abs()
        idx = diffs.idxmin()
        if diffs.loc[idx] <= tol_years:
            r_ipca = curva_ipca.loc[idx]
            taxa_pre = p["taxa_atual"]
            taxa_real = r_ipca["taxa_atual"]
            # normaliza se estiver em %
            tp = taxa_pre / 100.0 if taxa_pre > 1 else taxa_pre
            tr = taxa_real / 100.0 if taxa_real > 1 else taxa_real
            implied = (1 + tp) / (1 + tr) - 1.0
            pairs.append(
                {
                    "pre_titulo": p["titulo"],
                    "pre_vencimento": p["vencimento"].date(),
                    "ipca_titulo": r_ipca["titulo"],
                    "ipca_vencimento": r_ipca["vencimento"].date(),
                    "prazo_medio_a": (p["anos_ate_venc"] + r_ipca["anos_ate_venc"])
                    / 2,
                    "taxa_pre_%a.a.": tp * 100,
                    "taxa_real_%a.a.": tr * 100,
                    "inflacao_implicita_%a.a.": implied * 100,
                }
            )

    df_pairs = pd.DataFrame(pairs)
    if df_pairs.empty:
        return df_pairs
    return df_pairs.sort_values("prazo_medio_a").reset_index(drop=True)


# ===================== ETTJ ANBIMA (opcional) =====================
def fetch_anbima_ettj(curva: str = "PRE") -> pd.DataFrame | None:
    """
    Tenta buscar a ETTJ ANBIMA via biblioteca `pyettj`, se disponível.
    Retorna None em caso de falha (rede, parsing, libs ausentes).

    curva: "PRE" ou "IPCA".
    """
    try:
        import pyettj.ettj as ettj_lib  # type: ignore
    except Exception:
        return None

    try:
        # tenta nos últimos 5 dias úteis (ANBIMA só disponibiliza D-5)
        for offset in range(0, 7):
            d = (pd.Timestamp.today() - pd.tseries.offsets.BDay(offset)).strftime(
                "%d/%m/%Y"
            )
            try:
                curv = curva.upper()
                # pyettj aceita "PRE" / "IPCA" / "TR" / "EUR" / "JPY" / "USD"
                df_e = ettj_lib.get_ettj(d, curva=curv)
                if df_e is not None and not df_e.empty:
                    df_e = df_e.copy()
                    df_e["ref_date"] = d
                    return df_e
            except Exception:
                continue
    except Exception:
        return None
    return None


# ===================== Carrego IPCA+ =====================
def ir_aliquota_por_prazo(dias: int) -> float:
    """Tabela regressiva do IR para renda fixa."""
    if dias <= 180:
        return 0.225
    if dias <= 360:
        return 0.20
    if dias <= 720:
        return 0.175
    return 0.15


def simulate_carry_ipca(
    aporte: float,
    real_rate_a: float,
    ipca_rate_a: float,
    start_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    freq: str = "ME",
    include_fees: bool = True,
    custody_fee_a: float = 0.002,
    apply_ir: bool = True,
    periodic_contribution: float = 0.0,
    contribution_freq: str = "ME",
) -> pd.DataFrame:
    """
    Cenário de CARREGO até o vencimento (IPCA+).

    Parâmetros:
    - aporte: aporte inicial em R$.
    - real_rate_a: taxa real contratada (decimal, ex.: 0.0793).
    - ipca_rate_a: IPCA assumido (decimal, ex.: 0.04).
    - start_date / maturity_date.
    - freq: frequência da série retornada ("ME", "D", ...).
    - include_fees / custody_fee_a: custódia anualizada.
    - apply_ir: aplica IR regressivo no resgate final.
    - periodic_contribution: aporte adicional periódico em R$ (default 0 = DCA off).
    - contribution_freq: frequência dos aportes extras ("ME" mensal, "YE" anual).

    Retorna DataFrame com valor_bruto, valor_pos_fees, (se apply_ir) valor_liquido_final_ir.
    Quando há aportes periódicos, a coluna `aportado_acumulado` aparece.
    """
    # compat pandas 2.2+
    _freq_compat = {"M": "ME", "A": "YE", "Y": "YE"}
    freq = _freq_compat.get(freq, freq)
    contribution_freq = _freq_compat.get(contribution_freq, contribution_freq)

    start_date = pd.Timestamp(start_date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()

    if maturity_date <= start_date:
        raise ValueError("maturity_date deve ser maior que start_date")

    dates = pd.date_range(start=start_date, end=maturity_date, freq=freq)
    if len(dates) == 0 or dates[-1] != maturity_date:
        dates = dates.append(pd.DatetimeIndex([maturity_date]))

    nominal_rate_a = (1 + real_rate_a) * (1 + ipca_rate_a) - 1.0
    fee_a = custody_fee_a if include_fees else 0.0

    # Taxa líquida (depois da custódia) — para compor o saldo
    net_rate_a = (1 + nominal_rate_a) * (1 - fee_a) - 1.0  # aprox
    # Para compatibilidade com a versão anterior (sem aportes):
    # usamos (1+nominal)^t * (1-fee)^t, equivalente a (1+net)^t no limite
    # mas aqui precisamos de evolução correta com aportes intermediários.

    # Se NÃO há aportes periódicos: usa fórmula fechada (idêntico à versão anterior)
    if periodic_contribution == 0.0:
        t_years = (dates - start_date).days / 365.25
        valor_bruto = (
            aporte * (1 + real_rate_a) ** t_years * (1 + ipca_rate_a) ** t_years
        )
        valor_pos_fees = (
            valor_bruto * (1 - fee_a) ** t_years if fee_a > 0 else valor_bruto
        )
        df = pd.DataFrame(
            {
                "data": dates,
                "anos": t_years,
                "aportado_acumulado": aporte,
                "valor_bruto": valor_bruto,
                "valor_pos_fees": valor_pos_fees,
            }
        )
    else:
        # Com aportes periódicos: evolução passo a passo
        contribution_dates = pd.date_range(
            start=start_date, end=maturity_date, freq=contribution_freq
        )
        contribution_dates = set(pd.DatetimeIndex(contribution_dates).normalize())

        saldo = aporte
        aportado = aporte
        rows = []
        for i, d in enumerate(dates):
            if i == 0:
                dt_years = 0.0
            else:
                dt_years = (d - dates[i - 1]).days / 365.25
            # cresce o saldo pelo período
            saldo = saldo * (1 + net_rate_a) ** dt_years
            # se houve data de aporte periódico dentro deste passo, adiciona
            # (aproximação: adiciona no fim do passo)
            in_step_contrib = 0.0
            if i > 0:
                in_step = [
                    dd
                    for dd in contribution_dates
                    if dates[i - 1] < dd <= d and dd != start_date
                ]
                in_step_contrib = periodic_contribution * len(in_step)
            saldo += in_step_contrib
            aportado += in_step_contrib
            t_years = (d - start_date).days / 365.25
            rows.append(
                {
                    "data": d,
                    "anos": t_years,
                    "aportado_acumulado": aportado,
                    "valor_bruto": saldo / (1 - fee_a) ** t_years
                    if fee_a > 0
                    else saldo,
                    "valor_pos_fees": saldo,
                }
            )
        df = pd.DataFrame(rows)

    if apply_ir:
        dias = int((maturity_date - start_date).days)
        aliquota = ir_aliquota_por_prazo(dias)
        total_aportado = float(df["aportado_acumulado"].iloc[-1])
        ganho_nominal = float(df["valor_pos_fees"].iloc[-1] - total_aportado)
        imposto = max(0.0, ganho_nominal) * aliquota

        df["aliquota_ir"] = aliquota
        df["imposto_ir_final"] = 0.0
        df.loc[df.index[-1], "imposto_ir_final"] = imposto
        df["valor_liquido_final_ir"] = df["valor_pos_fees"]
        df.loc[df.index[-1], "valor_liquido_final_ir"] = float(
            df["valor_pos_fees"].iloc[-1] - imposto
        )

    return df


# ===================== Comparação Carrego vs Venda antecipada (MtM) =====================
def simulate_sell_before_maturity(
    series: pd.DataFrame,
    titulo: str,
    aporte: float,
    sell_years: float,
    taxa_future: float,
    ipca_rate_a: float = 0.04,
    lookback_days: int = 730,
    custody_fee_a: float = 0.002,
    include_fees: bool = True,
) -> dict:
    """
    Compara dois cenários ao longo dos próximos `sell_years` anos:

    1) CARREGO: segura até o vencimento, recebendo taxa_real * IPCA.
    2) VENDA ANTECIPADA (MtM): vende o título em `sell_years` anos,
       assumindo que a taxa naquele momento será `taxa_future`. PU
       estimado via regressão PU~taxa.

    Para o caso 1, calcula o valor acumulado até a data de venda (não até o
    vencimento — é comparação justa: mesmo horizonte de tempo).

    Observação: IR considerado regressivo, aplicado sobre o ganho nominal.

    Retorna dict com valor_carrego, valor_venda, diferenca.
    """
    taxa_real_a = float(series["taxa"].iloc[-1])
    taxa_real_a = taxa_real_a / 100.0 if taxa_real_a > 1 else taxa_real_a

    pu_atual = float(series["pu"].iloc[-1])
    units = aporte / pu_atual if pu_atual > 0 else 0

    # ---- Venda antecipada ----
    est = estimate_pu_from_taxa(series, taxa_future, lookback_days=lookback_days)
    pu_venda = est["pu_est"]

    # Aplica crescimento do VNA (marcação mantém a indexação do IPCA mesmo vendendo)
    # PU estimado já reflete a taxa futura, mas o VNA cresce com IPCA.
    # Aproximação: PU estimado é na data de hoje com taxa futura;
    # aplicamos o crescimento do VNA no período.
    vna_fator = (1 + ipca_rate_a) ** sell_years
    valor_bruto_venda = units * pu_venda * vna_fator

    # custódia
    if include_fees and custody_fee_a > 0:
        valor_pos_fees_venda = valor_bruto_venda * (1 - custody_fee_a) ** sell_years
    else:
        valor_pos_fees_venda = valor_bruto_venda

    # IR regressivo (aplica sobre o ganho)
    dias_venda = int(sell_years * 365.25)
    aliq_venda = ir_aliquota_por_prazo(dias_venda)
    ganho_venda = max(0.0, valor_pos_fees_venda - aporte)
    valor_liq_venda = valor_pos_fees_venda - ganho_venda * aliq_venda

    # ---- Carrego até o mesmo horizonte (sell_years) ----
    # Mantém até `sell_years` anos, depois resgata.
    valor_bruto_carry = (
        aporte * (1 + taxa_real_a) ** sell_years * (1 + ipca_rate_a) ** sell_years
    )
    if include_fees and custody_fee_a > 0:
        valor_pos_fees_carry = valor_bruto_carry * (1 - custody_fee_a) ** sell_years
    else:
        valor_pos_fees_carry = valor_bruto_carry
    ganho_carry = max(0.0, valor_pos_fees_carry - aporte)
    valor_liq_carry = valor_pos_fees_carry - ganho_carry * aliq_venda

    return {
        "titulo": titulo,
        "sell_years": sell_years,
        "taxa_future_pressuposta": taxa_future,
        "pu_atual": pu_atual,
        "pu_venda_estimado": pu_venda,
        "valor_venda_bruto": valor_bruto_venda,
        "valor_venda_liquido": valor_liq_venda,
        "valor_carry_bruto": valor_bruto_carry,
        "valor_carry_liquido": valor_liq_carry,
        "diferenca_liquida": valor_liq_venda - valor_liq_carry,
        "vantagem": "venda" if valor_liq_venda > valor_liq_carry else "carrego",
        "aliquota_ir_usada": aliq_venda,
    }


# ===================== Fat Tail Index (Taleb / MAD-STD) =====================
# Referência teórica: Taleb, "Statistical Consequences of Fat Tails" (cap 4.4.1).
# Para uma distribuição Gaussiana pura, STD/MAD = sqrt(pi/2) ≈ 1.2533
# (equivalente: MAD/STD = sqrt(2/pi) ≈ 0.7979).
#
# Convenção de Taleb:  iFat_taleb = STD / MAD (gaussiano ≈ 1.253, acima = cauda gorda)
# Convenção MAD/STD :  iFat_mad   = MAD / STD (gaussiano ≈ 0.798, abaixo = cauda gorda)
#
# As duas são inversas. Mantemos ambas para compatibilidade com literatura e estratégias.

GAUSSIAN_STD_OVER_MAD = float(np.sqrt(np.pi / 2.0))   # ≈ 1.2533
GAUSSIAN_MAD_OVER_STD = float(np.sqrt(2.0 / np.pi))   # ≈ 0.7979


def _mad(x) -> float:
    """Mean Absolute Deviation em torno da média."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x - np.mean(x))))


def compute_fat_tail_index(
    series: pd.DataFrame,
    window: int = 60,
    price_col: str = "pu",
    convention: str = "mad_over_std",
) -> pd.DataFrame:
    """
    Calcula o índice de cauda gorda (Fat Tail Index) em janela móvel sobre
    os retornos do preço (PU).

    Parâmetros:
    - series: DataFrame com colunas 'data' e `price_col` (ex.: 'pu' ou 'taxa').
    - window: tamanho da janela móvel (default 60 pregões ≈ 3 meses).
    - price_col: coluna para calcular os retornos (default 'pu').
    - convention:
        "mad_over_std" (default, notebook original) → iFat = MAD/STD.
            Gaussiano ≈ 0.7979. Valores **abaixo** = caudas gordas.
        "std_over_mad" (Taleb) → iFat = STD/MAD.
            Gaussiano ≈ 1.2533. Valores **acima** = caudas gordas.

    Retorna DataFrame com colunas:
        data, {price_col}, retorno, MAD, STD, iFat,
        gaussian_ref, iFat_ma (média móvel do próprio iFat),
        iFat_ma_minus_sd (banda inferior do iFat, usada como gatilho),
        is_fat_tail (bool: True quando está em zona de cauda gorda).
    """
    if price_col not in series.columns:
        raise KeyError(
            f"Coluna '{price_col}' não existe em series. Disponíveis: {list(series.columns)}"
        )

    s = series[["data", price_col]].dropna().sort_values("data").copy()
    s["retorno"] = s[price_col].pct_change(1)
    s = s.dropna(subset=["retorno"]).reset_index(drop=True)

    if len(s) < window:
        # Sem dados suficientes — retorna vazio com colunas definidas
        cols = [
            "data",
            price_col,
            "retorno",
            "MAD",
            "STD",
            "iFat",
            "gaussian_ref",
            "iFat_ma",
            "iFat_ma_minus_sd",
            "is_fat_tail",
        ]
        return pd.DataFrame(columns=cols)

    s["MAD"] = s["retorno"].rolling(window).apply(_mad, raw=False)
    s["STD"] = s["retorno"].rolling(window).std(ddof=0)

    if convention == "std_over_mad":
        s["iFat"] = s["STD"] / s["MAD"]
        gaussian = GAUSSIAN_STD_OVER_MAD
        # Na convenção Taleb, "cauda gorda" é ACIMA do gaussiano
        # Mantemos o mesmo contrato de "is_fat_tail" (True = tail)
        # via comparação com média móvel + banda
        s["iFat_ma"] = s["iFat"].rolling(window).mean()
        s["iFat_ma_minus_sd"] = (
            s["iFat"].rolling(window).mean() + s["iFat"].rolling(window).std(ddof=0)
        )
        s["is_fat_tail"] = s["iFat"] > s["iFat_ma_minus_sd"]
    else:
        # MAD/STD (notebook original)
        s["iFat"] = s["MAD"] / s["STD"]
        gaussian = GAUSSIAN_MAD_OVER_STD
        s["iFat_ma"] = s["iFat"].rolling(window).mean()
        s["iFat_ma_minus_sd"] = (
            s["iFat"].rolling(window).mean() - s["iFat"].rolling(window).std(ddof=0)
        )
        # cauda gorda: valor baixo (abaixo da banda inferior)
        s["is_fat_tail"] = s["iFat"] < s["iFat_ma_minus_sd"]

    s["gaussian_ref"] = gaussian
    s["convention"] = convention
    s["window"] = window
    return s.dropna(subset=["iFat"]).reset_index(drop=True)


def fat_tail_summary(ft_df: pd.DataFrame) -> dict:
    """Resumo descritivo do iFat (média, mediana, quartis, % em cauda gorda)."""
    if ft_df.empty or "iFat" not in ft_df.columns:
        return {}
    x = ft_df["iFat"].dropna()
    if len(x) == 0:
        return {}
    convention = (
        ft_df["convention"].iloc[0]
        if "convention" in ft_df.columns
        else "mad_over_std"
    )
    return {
        "convention": convention,
        "n_observacoes": int(len(x)),
        "media": float(x.mean()),
        "mediana": float(x.median()),
        "desvio": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
        "gaussian_ref": float(ft_df["gaussian_ref"].iloc[-1]),
        "valor_atual": float(x.iloc[-1]),
        "pct_em_cauda_gorda": float(ft_df["is_fat_tail"].mean()),
    }


def fat_tail_entry_signals(
    ft_df: pd.DataFrame,
    absolute_threshold: float | None = 0.65,
    use_moving_band: bool = True,
) -> pd.DataFrame:
    """
    Gera os sinais de "zona de entrada" combinando:
      - iFat < banda móvel (iFat_ma_minus_sd), E
      - iFat < absolute_threshold (ex: 0.65 na convenção MAD/STD).

    Adapta os comparadores automaticamente à convenção usada.
    Retorna DataFrame com coluna booleana `entrada`.
    """
    if ft_df.empty:
        return ft_df.assign(entrada=False)

    out = ft_df.copy()
    convention = out["convention"].iloc[0] if "convention" in out.columns else "mad_over_std"

    if convention == "std_over_mad":
        # caudas gordas → iFat ALTO; zona de "entrada" = iFat muito alto
        band_cond = (out["iFat"] > out["iFat_ma_minus_sd"]) if use_moving_band else True
        # threshold default "aproximado" simétrico ao 0.65 do original
        # (0.65 é ~20% abaixo do gaussiano 0.798; 20% acima de 1.253 é ~1.503)
        if absolute_threshold is None:
            abs_cond = True
        else:
            # usuário passou 0.65 pensando em MAD/STD → convertemos
            # se ele passou valor > 1, assume que já é na convenção Taleb
            thr = absolute_threshold if absolute_threshold > 1 else 1.0 / absolute_threshold
            abs_cond = out["iFat"] > thr
        out["entrada"] = band_cond & abs_cond
    else:
        # MAD/STD → caudas gordas → iFat BAIXO
        band_cond = (out["iFat"] < out["iFat_ma_minus_sd"]) if use_moving_band else True
        abs_cond = (out["iFat"] < absolute_threshold) if absolute_threshold is not None else True
        out["entrada"] = band_cond & abs_cond

    return out


def backtest_fat_tail_strategy(
    series: pd.DataFrame,
    window: int = 60,
    price_col: str = "pu",
    convention: str = "mad_over_std",
    absolute_threshold: float | None = 0.65,
    use_moving_band: bool = True,
    horizons=(30, 60, 90, 180),
) -> dict:
    """
    Backtest: quando o índice Fat Tail marca "zona de entrada", medimos o
    retorno do PU em horizontes à frente. Serve pra validar se a estratégia
    "comprar quando iFat sinaliza fat tail" gera ganho assimétrico.

    Retorna:
      - events: DataFrame com as datas de sinal e retornos forward.
      - summary: dicionário com estatísticas por horizonte.
    """
    ft = compute_fat_tail_index(
        series, window=window, price_col=price_col, convention=convention
    )
    ft = fat_tail_entry_signals(
        ft, absolute_threshold=absolute_threshold, use_moving_band=use_moving_band
    )

    if ft.empty:
        return {"events": pd.DataFrame(), "summary": {"n_events": 0}, "series": ft}

    # calcula retornos forward do preço
    fwd = forward_change(
        ft[["data", price_col]].rename(columns={price_col: "pu"}),
        "pu",
        horizons=horizons,
    )

    merged = ft.merge(fwd.drop(columns=["pu"]), on="data", how="left")
    events = merged[merged["entrada"]].copy()

    summary = {
        "n_events": int(len(events)),
        "window": window,
        "convention": convention,
        "absolute_threshold": absolute_threshold,
        "use_moving_band": use_moving_band,
        "total_obs": int(len(ft)),
        "pct_em_entrada": float(ft["entrada"].mean()),
    }

    for h in horizons:
        col = f"ret_pu_{h}d"
        if col in events.columns:
            vals = events[col].dropna()
            if len(vals) > 0:
                summary[f"mean_ret_{h}d"] = float(vals.mean())
                summary[f"median_ret_{h}d"] = float(vals.median())
                summary[f"p25_ret_{h}d"] = float(vals.quantile(0.25))
                summary[f"p75_ret_{h}d"] = float(vals.quantile(0.75))
                summary[f"hit_rate_{h}d"] = float((vals > 0).mean())
                std_v = float(vals.std(ddof=0))
                summary[f"sharpe_like_{h}d"] = (
                    float(vals.mean() / std_v) if std_v > 0 else np.nan
                )
                # compara com retorno médio "base" (sem filtro) no mesmo horizonte
                base_vals = merged[col].dropna()
                if len(base_vals) > 0:
                    summary[f"baseline_mean_{h}d"] = float(base_vals.mean())
                    summary[f"excesso_vs_baseline_{h}d"] = float(
                        vals.mean() - base_vals.mean()
                    )

    return {"events": events, "summary": summary, "series": ft}


def compute_fat_tail_current(
    df: pd.DataFrame,
    window: int = 60,
    convention: str = "mad_over_std",
    absolute_threshold: float | None = 0.65,
) -> pd.DataFrame:
    """
    Varre TODOS os pares (título, vencimento) e devolve o valor ATUAL do iFat
    + flag de "em zona de entrada". Útil para integrar no Scanner.

    Retorna: DataFrame com titulo, vencimento, iFat_atual, gaussian_ref,
             em_zona_entrada, pct_historico_em_entrada.
    """
    rows = []
    for (titulo, vencimento), g in df.groupby(["titulo", "vencimento"]):
        if "pu" not in g.columns:
            continue
        try:
            ft = compute_fat_tail_index(
                g, window=window, price_col="pu", convention=convention
            )
            if ft.empty:
                continue
            ft = fat_tail_entry_signals(
                ft, absolute_threshold=absolute_threshold, use_moving_band=True
            )
            last = ft.iloc[-1]
            rows.append(
                {
                    "titulo": titulo,
                    "vencimento": vencimento.date(),
                    "iFat_atual": float(last["iFat"]),
                    "iFat_ma": float(last["iFat_ma"]) if pd.notna(last["iFat_ma"]) else np.nan,
                    "gaussian_ref": float(last["gaussian_ref"]),
                    "em_zona_entrada": bool(last["entrada"]),
                    "pct_historico_em_entrada": float(ft["entrada"].mean()),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


# ===================== Fat Tail — Position Sizing Escalonado =====================
# Abordagem motivada pela observação empírica: o sinal binário (entrada sim/não)
# com banda móvel ATRASA — quando a banda finalmente permite a entrada, o PU já
# reagiu. A solução é:
#   1) usar o iFat absoluto em vez de depender da banda móvel (remove lag);
#   2) escalonar o tamanho da posição conforme a profundidade do stress.
#
# Interpretação: em vez de "comprar tudo ou nada quando dispara", você começa a
# comprar quando o iFat mostra pressão (0.75-0.80) e vai aumentando a posição
# conforme cai. Isso melhora o preço médio e protege contra falsos sinais.

# Níveis default (convenção MAD/STD — gaussiano ≈ 0.7979)
# Observação: 0.7979 é o valor gaussiano "normal". Só valores claramente ABAIXO
# indicam stress real. Os níveis começam em 0.75 (já abaixo do gaussiano) para
# evitar entrar em flutuações normais.
DEFAULT_SIZING_LEVELS_MAD = [
    (0.75, 0.00),  # >= 0.75: sem posição (zona normal)
    (0.70, 0.25),  # 0.70-0.75: leve pressão, começa entrada
    (0.65, 0.50),  # 0.65-0.70: stress moderado
    (0.60, 0.75),  # 0.60-0.65: stress forte
    (0.00, 1.00),  # < 0.60: pânico, posição cheia
]

# Níveis equivalentes na convenção STD/MAD (Taleb, valores aprox. reciprocos)
DEFAULT_SIZING_LEVELS_STD = [
    (1.33, 0.00),  # <= 1.33: sem posição
    (1.43, 0.25),
    (1.54, 0.50),
    (1.67, 0.75),
    (np.inf, 1.00),
]


def position_size_from_ifat(
    ifat: float,
    convention: str = "mad_over_std",
    levels: list[tuple[float, float]] | None = None,
) -> float:
    """
    Converte o valor atual do iFat em % da posição total a ser alocada.

    Parâmetros:
    - ifat: valor atual do índice fat tail.
    - convention: "mad_over_std" (default) ou "std_over_mad".
    - levels: lista de (threshold, size) customizada. Se None, usa default.
        Na convenção MAD/STD, os thresholds são ordenados do MAIOR para o MENOR
        (vai entrando em posição conforme o iFat CAI).
        Na convenção STD/MAD, é o contrário.

    Retorna: fração da posição [0.0 a 1.0].

    Exemplo (MAD/STD):
        iFat = 0.82 → 0% (mercado normal)
        iFat = 0.78 → 25% (leve pressão)
        iFat = 0.72 → 50% (stress moderado)
        iFat = 0.67 → 75% (stress forte)
        iFat = 0.50 → 100% (pânico)
    """
    if pd.isna(ifat):
        return 0.0

    if levels is None:
        levels = (
            DEFAULT_SIZING_LEVELS_MAD
            if convention == "mad_over_std"
            else DEFAULT_SIZING_LEVELS_STD
        )

    if convention == "mad_over_std":
        # ifat CAI → posição aumenta. Thresholds ordenados do maior p/ menor.
        for thr, size in levels:
            if ifat >= thr:
                return float(size)
        return float(levels[-1][1])
    else:
        # std/mad: ifat SOBE → posição aumenta.
        for thr, size in levels:
            if ifat <= thr:
                return float(size)
        return float(levels[-1][1])


def compute_fat_tail_with_sizing(
    series: pd.DataFrame,
    window: int = 60,
    price_col: str = "pu",
    convention: str = "mad_over_std",
    levels: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Calcula o iFat + tamanho de posição escalonado em cada data.

    Retorna DataFrame com colunas: data, pu (ou price_col), iFat, posicao_target,
    posicao_change (variação vs dia anterior, positivo = compra, negativo = venda).
    """
    ft = compute_fat_tail_index(
        series, window=window, price_col=price_col, convention=convention
    )
    if ft.empty:
        return ft

    ft = ft.copy()
    ft["posicao_target"] = ft["iFat"].apply(
        lambda x: position_size_from_ifat(x, convention=convention, levels=levels)
    )
    # mudança em cada dia (delta de posição)
    ft["posicao_change"] = ft["posicao_target"].diff().fillna(ft["posicao_target"].iloc[0])
    return ft


def backtest_fat_tail_sizing(
    series: pd.DataFrame,
    window: int = 60,
    price_col: str = "pu",
    convention: str = "mad_over_std",
    levels: list[tuple[float, float]] | None = None,
    capital_total: float = 100000.0,
    hold_horizons=(30, 60, 90, 180, 365),
) -> dict:
    """
    Backtest do esquema de position sizing escalonado.

    Lógica simulada:
    - Você tem `capital_total` disponível (ex.: R$ 100k).
    - A cada dia, olha o iFat e calcula a posição TARGET (0% a 100%).
    - Se a posição target aumentou em relação ao dia anterior → você COMPRA a
      diferença (em R$, ao PU daquele dia).
    - Se diminuiu → VENDE a diferença.
    - Mede o retorno acumulado da posição em diferentes horizontes após cada
      COMPRA (momento em que a posição target aumentou).

    Compara contra:
    - Comprar 100% num único dia quando o sinal binário tradicional dispara
      (estratégia da aba atual).
    - Buy-and-hold (comprar e manter sempre).

    Retorna dict com:
      - series_ft: DataFrame com iFat e posição ao longo do tempo
      - trades: DataFrame com cada compra/venda realizada
      - summary: estatísticas comparativas
    """
    ft = compute_fat_tail_with_sizing(
        series, window=window, price_col=price_col, convention=convention, levels=levels
    )

    if ft.empty:
        return {"series_ft": ft, "trades": pd.DataFrame(), "summary": {}}

    # Identifica dias de COMPRA (posição aumentou)
    compras = ft[ft["posicao_change"] > 1e-9].copy()

    if compras.empty:
        return {
            "series_ft": ft,
            "trades": pd.DataFrame(),
            "summary": {"n_compras": 0},
        }

    # Para cada dia de compra, mede retorno do PU em cada horizonte
    ft_sorted = ft.sort_values("data").reset_index(drop=True)
    for h in hold_horizons:
        target = compras["data"] + pd.to_timedelta(h, unit="D")
        fwd_pu = []
        for td in target:
            j = ft_sorted["data"].searchsorted(td, side="left")
            fwd_pu.append(
                ft_sorted.iloc[j][price_col] if j < len(ft_sorted) else np.nan
            )
        compras[f"pu_fwd_{h}d"] = fwd_pu
        compras[f"ret_{h}d"] = np.array(fwd_pu) / compras[price_col].values - 1.0

    # Resumo por horizonte
    summary = {
        "n_compras": int(len(compras)),
        "convention": convention,
        "window": window,
        "capital_total": capital_total,
        "iFat_atual": float(ft["iFat"].iloc[-1]),
        "posicao_atual_target_%": float(ft["posicao_target"].iloc[-1] * 100),
    }

    for h in hold_horizons:
        col = f"ret_{h}d"
        vals = compras[col].dropna()
        if len(vals) == 0:
            continue
        # retorno médio PONDERADO pelo tamanho da compra (delta de posição)
        pesos = compras.loc[vals.index, "posicao_change"].values
        retornos = vals.values
        if pesos.sum() > 0:
            ret_ponderado = float(np.sum(retornos * pesos) / np.sum(pesos))
        else:
            ret_ponderado = float(np.mean(retornos))

        summary[f"ret_medio_ponderado_{h}d_%"] = round(ret_ponderado * 100, 3)
        summary[f"ret_medio_simples_{h}d_%"] = round(float(np.mean(retornos)) * 100, 3)
        summary[f"hit_rate_{h}d_%"] = round(float(np.mean(retornos > 0)) * 100, 1)
        summary[f"n_obs_{h}d"] = int(len(vals))

    # Baseline (comprar em qualquer dia)
    for h in hold_horizons:
        target = ft_sorted["data"] + pd.to_timedelta(h, unit="D")
        fwd_pu = []
        for td in target:
            j = ft_sorted["data"].searchsorted(td, side="left")
            fwd_pu.append(
                ft_sorted.iloc[j][price_col] if j < len(ft_sorted) else np.nan
            )
        base_ret = np.array(fwd_pu) / ft_sorted[price_col].values - 1.0
        base_ret = base_ret[~np.isnan(base_ret)]
        if len(base_ret) > 0:
            summary[f"baseline_{h}d_%"] = round(float(np.mean(base_ret)) * 100, 3)
            summary[f"excesso_vs_baseline_{h}d_%"] = round(
                (ret_ponderado - float(np.mean(base_ret))) * 100, 3
            )

    # Simulação de patrimônio: evolução da carteira ao longo do tempo
    # Lógica simplificada: dinheiro restante rende 0 (CDI/Selic poderia ser modelado depois).
    caixa = capital_total
    qtd_titulo = 0.0
    patrimonio = []
    for i, r in ft_sorted.iterrows():
        delta_pos = r["posicao_change"]
        if delta_pos > 0:
            # compra: transforma (delta_pos * capital_total) em PU
            valor_compra = delta_pos * capital_total
            valor_compra = min(valor_compra, caixa)
            qtd_compra = valor_compra / r[price_col] if r[price_col] > 0 else 0
            qtd_titulo += qtd_compra
            caixa -= valor_compra
        elif delta_pos < 0:
            # venda: libera parte do título em caixa
            frac_venda = -delta_pos  # ex.: 0.25 significa vender 25% do total
            qtd_venda = qtd_titulo * (
                -delta_pos / r["posicao_target"] if r["posicao_target"] > 0 else 1
            )
            # versão simplificada: vende proporcionalmente ao tamanho atual
            qtd_venda = min(qtd_venda, qtd_titulo)
            valor_venda = qtd_venda * r[price_col]
            qtd_titulo -= qtd_venda
            caixa += valor_venda

        patrimonio.append(caixa + qtd_titulo * r[price_col])

    ft_sorted["patrimonio"] = patrimonio
    ft_sorted["retorno_acum_%"] = (
        ft_sorted["patrimonio"] / capital_total - 1.0
    ) * 100

    # Compara com buy-and-hold (comprou tudo no primeiro dia)
    pu_inicial = float(ft_sorted[price_col].iloc[0])
    qtd_bh = capital_total / pu_inicial
    ft_sorted["patrimonio_bh"] = qtd_bh * ft_sorted[price_col]
    ft_sorted["retorno_bh_%"] = (
        ft_sorted["patrimonio_bh"] / capital_total - 1.0
    ) * 100

    summary["ret_final_escalonado_%"] = round(
        float(ft_sorted["retorno_acum_%"].iloc[-1]), 2
    )
    summary["ret_final_buy_hold_%"] = round(
        float(ft_sorted["retorno_bh_%"].iloc[-1]), 2
    )
    summary["diferenca_%"] = round(
        summary["ret_final_escalonado_%"] - summary["ret_final_buy_hold_%"], 2
    )

    return {
        "series_ft": ft_sorted,
        "trades": compras,
        "summary": summary,
    }


# ===================== Síntese final: Tabela de Oportunidades =====================
def compute_opportunities_table(
    df: pd.DataFrame,
    window: int = 90,
    convention: str = "mad_over_std",
    absolute_threshold: float = 0.65,
    sizing_levels: list[tuple[float, float]] | None = None,
    main_horizon: int = 90,
    min_events_for_reliability: int = 50,
) -> pd.DataFrame:
    """
    Síntese final de decisão de alocação.

    Para cada par (título, vencimento) com PU disponível, computa:
      - **Histórico do sinal** (backtest do iFat):
          excesso vs baseline no horizonte principal (default 90d)
          hit rate histórico
          número de eventos disparados
      - **Status atual**:
          iFat atual
          posição-alvo (via position sizing escalonado)
      - **Veredito automático**:
          "FORTE" / "OK" / "FRACO" / "EVITAR" / "SEM DADOS"
          com explicação textual

    Critérios do veredito (aplicados em cascata):
      - EVITAR: se excesso histórico < -0.5% OU hit rate < 45%
      - SEM DADOS: se n_eventos < 10
      - FORTE: excesso > 2%, hit rate > 60%, n_eventos >= min_events_for_reliability
      - OK: excesso > 0.5%, hit rate > 55%
      - FRACO: qualquer outro caso (excesso positivo mas pequeno)

    Retorna DataFrame ordenado com os melhores primeiro.
    """
    rows = []

    for (titulo, vencimento), g in df.groupby(["titulo", "vencimento"]):
        if "pu" not in g.columns or g["pu"].isna().all():
            continue

        # Só títulos com vencimento futuro
        if vencimento <= pd.Timestamp.today():
            continue

        anos_ate = (vencimento - pd.Timestamp.today().normalize()).days / 365.25

        try:
            # === Histórico: backtest do sinal ===
            ft = compute_fat_tail_index(g, window=window, price_col="pu", convention=convention)
            if ft.empty:
                continue
            ft = fat_tail_entry_signals(ft, absolute_threshold=absolute_threshold, use_moving_band=True)

            # retornos forward
            ft_sorted = ft.sort_values("data").reset_index(drop=True)
            target = ft_sorted["data"] + pd.to_timedelta(main_horizon, unit="D")
            fwd = []
            for td in target:
                j = ft_sorted["data"].searchsorted(td, side="left")
                fwd.append(ft_sorted.iloc[j]["pu"] if j < len(ft_sorted) else np.nan)
            ft_sorted[f"ret_{main_horizon}d"] = np.array(fwd) / ft_sorted["pu"] - 1.0

            eventos_mask = ft_sorted["entrada"]
            ev_rets = ft_sorted.loc[eventos_mask, f"ret_{main_horizon}d"].dropna()
            base_rets = ft_sorted[f"ret_{main_horizon}d"].dropna()

            if len(base_rets) == 0:
                continue

            n_eventos = len(ev_rets)
            baseline_mean = float(base_rets.mean())

            if n_eventos > 0:
                excesso = float(ev_rets.mean() - baseline_mean) * 100  # em %
                hit_rate = float((ev_rets > 0).mean()) * 100
            else:
                excesso = np.nan
                hit_rate = np.nan

            # === Status atual ===
            ifat_atual = float(ft["iFat"].iloc[-1]) if not ft.empty else np.nan
            pos_target = position_size_from_ifat(ifat_atual, convention=convention, levels=sizing_levels) * 100

            # === Veredito ===
            if n_eventos < 10:
                veredito = "📊 SEM DADOS"
                nota = f"Apenas {n_eventos} eventos históricos — insuficiente para validar."
                score = -100
            elif (not np.isnan(excesso)) and (excesso < -0.5 or (not np.isnan(hit_rate) and hit_rate < 45)):
                veredito = "❌ EVITAR"
                nota = f"Sinal historicamente NEGATIVO neste título (excesso {excesso:+.2f}%, hit {hit_rate:.0f}%)."
                score = -50 + excesso
            elif (not np.isnan(excesso)) and excesso > 2.0 and hit_rate > 60 and n_eventos >= min_events_for_reliability:
                veredito = "✅ FORTE"
                nota = f"Sinal muito confiável: {n_eventos} eventos, {hit_rate:.0f}% hit rate."
                score = excesso * 2 + hit_rate / 10
            elif (not np.isnan(excesso)) and excesso > 0.5 and hit_rate > 55:
                veredito = "🟢 OK"
                nota = f"Sinal útil ({n_eventos} eventos), porém com ganho mais moderado."
                score = excesso + hit_rate / 20
            else:
                veredito = "🟡 FRACO"
                nota = "Excesso pequeno ou hit rate mediano — use com cautela."
                score = (excesso or 0) / 2

            # === Recomendação final (combina veredito com posição atual) ===
            if veredito in ("❌ EVITAR", "📊 SEM DADOS"):
                recomendacao = "Não usar o sinal aqui"
            elif pos_target == 0:
                recomendacao = "Aguardar — iFat ainda em zona normal"
            else:
                recomendacao = f"Alocar {pos_target:.0f}% da posição reservada"

            rows.append({
                "titulo": titulo,
                "vencimento": vencimento.date(),
                "anos_ate_venc": round(anos_ate, 1),
                "iFat_atual": round(ifat_atual, 4) if not np.isnan(ifat_atual) else None,
                "posicao_alvo_%": round(pos_target, 0),
                f"excesso_{main_horizon}d_%": round(excesso, 2) if not np.isnan(excesso) else None,
                "hit_rate_%": round(hit_rate, 1) if not np.isnan(hit_rate) else None,
                "n_eventos_hist": n_eventos,
                "veredito": veredito,
                "recomendacao": recomendacao,
                "nota": nota,
                "_score": score,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("_score", ascending=False).reset_index(drop=True)
    return result.drop(columns=["_score"])
