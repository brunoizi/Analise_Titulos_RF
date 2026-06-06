"""
Scanner Quantitativo – Tesouro Direto (v2).

Abas:
1) 🏆 Scanner         – ranking J/Z + score.
2) 🧪 Backtest        – reversão à média verificável, com métricas de risco.
3) 📐 Risco           – duration, convexidade, DV01 e sensibilidade.
4) 🌐 Curvas & Inflação – curva de juros + inflação implícita (pré vs IPCA+).
5) 📈 Cenários (R$)   – PU futuro, carrego IPCA+ (com DCA), carrego vs venda antecipada.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Carrega variáveis do .env (API keys, etc)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv opcional

# Módulo de agentes de IA (aba Comitê) — opcional
try:
    from agents import (
        build_context_for_title,
        run_committee,
        ANTHROPIC_AVAILABLE,
    )
    AGENTS_MODULE_AVAILABLE = True
except Exception:
    AGENTS_MODULE_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False

from engine import (
    load_csv,
    prepare,
    compute_signals_all,
    get_series,
    backtest_mean_reversion,
    estimate_pu_from_taxa,
    simulate_carry_ipca,
    compute_risk_metrics,
    pu_sensitivity_table,
    build_yield_curve,
    implied_inflation,
    fetch_anbima_ettj,
    simulate_sell_before_maturity,
    classify_family,
    compute_fat_tail_index,
    fat_tail_summary,
    fat_tail_entry_signals,
    backtest_fat_tail_strategy,
    compute_fat_tail_current,
    GAUSSIAN_MAD_OVER_STD,
    GAUSSIAN_STD_OVER_MAD,
    position_size_from_ifat,
    compute_fat_tail_with_sizing,
    backtest_fat_tail_sizing,
    DEFAULT_SIZING_LEVELS_MAD,
    compute_opportunities_table,
)

# ---------- Page config ----------
st.set_page_config(page_title="Scanner Tesouro Direto", layout="wide")
st.title("📊 Scanner Quantitativo – Tesouro Direto")
st.caption(
    "Fonte: Tesouro Transparente (CSV oficial). "
    "Ferramenta quantitativa para estudo — não é recomendação de investimento."
)


# ---------- Helpers ----------
def download_btn(df: pd.DataFrame, filename: str, label: str = "⬇️ Baixar CSV"):
    """Botão de download para qualquer DataFrame."""
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )


def fmt_brl(x: float) -> str:
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)


# ---------- Sidebar ----------
st.sidebar.header("Fonte")
taxa_col = st.sidebar.selectbox(
    "Taxa utilizada",
    ["Taxa Compra Manha", "Taxa Compra Tarde", "Taxa Venda Manha", "Taxa Venda Tarde"],
    index=0,
)
pu_col = st.sidebar.selectbox(
    "PU utilizado (para backtest/cenários/risco)",
    ["PU Compra Manha", "PU Venda Manha", "PU Base Manha"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 **Dica**: para IPCA+ com Juros Semestrais e prefixados com cupom, "
    "a duration do app assume cupom de 10% a.a. (padrão Tesouro)."
)


# ---------- Cache ----------
@st.cache_data(ttl=60 * 60, show_spinner="Baixando dados do Tesouro Transparente...")
def load_all(taxa_col: str, pu_col: str) -> pd.DataFrame:
    raw = load_csv()
    return prepare(raw, taxa_col=taxa_col, pu_col=pu_col)


try:
    df = load_all(taxa_col, pu_col)
except Exception as e:
    st.error(f"Falha ao carregar dados: {e}")
    st.stop()


# ---------- Tabs ----------
tab1, tab_opp, tab2, tab3, tab4, tab5, tab6, tab_ia, tab7 = st.tabs(
    [
        "🏆 Scanner",
        "🎯 Oportunidades",
        "🧪 Backtest",
        "📐 Risco (Duration)",
        "🌐 Curvas & Inflação",
        "📈 Cenários (R$)",
        "🐘 Fat Tails",
        "🤖 Comitê",
        "📚 Guia",
    ]
)


# ===================== TAB 1: Scanner =====================
with tab1:
    st.subheader("🏆 Ranking (Scanner)")
    st.caption(
        f"Total de séries (título+vencimento): "
        f"{len(df.groupby(['titulo','vencimento'])):,}"
    )

    familias = st.multiselect(
        "Famílias",
        ["IPCA+", "Prefixado", "Educa+", "IGPM+", "Selic", "Renda+"],
        default=["IPCA+", "Prefixado", "Educa+"],
    )
    score_min = st.slider("Score mínimo", 0.0, 15.0, 0.0, 0.5)
    top_n = st.selectbox("Exibir", [20, 50, 100, 200, "Todos"], index=1)

    hoje = pd.Timestamp.today().normalize()
    sinais = compute_signals_all(df, windows_days=(180, 365, 730), min_points=80).copy()
    sinais["vencimento"] = pd.to_datetime(sinais["vencimento"])
    sinais["anos_ate_venc"] = (sinais["vencimento"] - hoje).dt.days / 365.25

    def bucket_prazo(x):
        if pd.isna(x):
            return "N/A"
        if x <= 3:
            return "Curto (<=3a)"
        if x <= 8:
            return "Médio (3–8a)"
        return "Longo (>8a)"

    sinais["bucket_prazo"] = sinais["anos_ate_venc"].apply(bucket_prazo)

    if familias:
        sinais = sinais[
            sinais["titulo"].str.contains("|".join(familias), case=False, na=False)
        ]
    sinais = sinais[sinais["score"] >= score_min]

    # ---- Filtro opcional Fat Tail ----
    st.markdown("#### 🐘 Filtro Fat Tail (opcional)")
    col_ft1, col_ft2, col_ft3 = st.columns([2, 1, 1])
    with col_ft1:
        use_fat_filter = st.checkbox(
            "Destacar/filtrar títulos em zona de cauda gorda",
            value=False,
            help=(
                "Calcula o índice MAD/STD dos retornos do PU em janela móvel. "
                "Caudas gordas (iFat baixo) indicam momentos de stress — "
                "frequentemente oportunidades de entrada."
            ),
        )
    with col_ft2:
        ft_window = st.selectbox(
            "Janela iFat (dias)", [30, 60, 90, 120], index=1, key="ft_window_scan"
        )
    with col_ft3:
        ft_threshold = st.number_input(
            "Limiar iFat",
            value=0.65,
            step=0.05,
            min_value=0.4,
            max_value=0.8,
            key="ft_thr_scan",
            help="Valor abaixo do qual o título entra em 'zona de entrada' (convenção MAD/STD, gaussiano≈0.798).",
        )

    if use_fat_filter:
        with st.spinner("Calculando índice Fat Tail para todos os títulos..."):
            ft_current = compute_fat_tail_current(
                df,
                window=int(ft_window),
                convention="mad_over_std",
                absolute_threshold=float(ft_threshold),
            )
        if not ft_current.empty:
            sinais = sinais.merge(
                ft_current[["titulo", "iFat_atual", "em_zona_entrada"]],
                left_on=["titulo"],
                right_on=["titulo"],
                how="left",
            )
            # se o merge bagunçou por duplicata (títulos repetidos com outros vencimentos),
            # fazemos pelo par (titulo,venc) se possível
            # (acima já pegou; aqui só garantimos que vencimento bate)
            # limpa duplicatas mantendo primeira ocorrência
            sinais = sinais.drop_duplicates(subset=["titulo", "vencimento"], keep="first")

            only_fat = st.checkbox("Mostrar APENAS títulos em zona de entrada", value=False)
            if only_fat:
                sinais = sinais[sinais["em_zona_entrada"] == True]

    sinais = sinais.sort_values("score", ascending=False)
    if top_n != "Todos":
        sinais = sinais.head(int(top_n))

    cols = [
        "titulo",
        "vencimento",
        "anos_ate_venc",
        "bucket_prazo",
        "taxa_atual",
        "J365",
        "Z365",
        "J730",
        "Z730",
        "concord_365_730",
        "score",
    ]
    if use_fat_filter and "iFat_atual" in sinais.columns:
        cols = cols + ["iFat_atual", "em_zona_entrada"]

    st.dataframe(sinais[cols], width="stretch", hide_index=True)
    download_btn(sinais[cols], "scanner_ranking.csv")

    st.subheader("📊 Top 25 por score")
    plot_df = sinais.head(25).copy()
    if not plot_df.empty:
        labels = plot_df.apply(
            lambda r: f"{r['titulo']} ({pd.to_datetime(r['vencimento']).date()})",
            axis=1,
        ).tolist()
        values = plot_df["score"].values
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(range(len(values)), values)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        st.pyplot(fig, clear_figure=True)


# ===================== TAB Oportunidades =====================
with tab_opp:
    st.subheader("🎯 Oportunidades — o que comprar AGORA")
    st.markdown(
        """
        **Síntese operacional do app.** Para cada título disponível, esta tela combina:

        1. **Histórico do sinal Fat Tail** (excesso vs baseline + hit rate)
        2. **Status atual** (iFat + posição-alvo escalonada)
        3. **Veredito automático** classificando onde vale a pena usar o sinal

        É a **tela de decisão** — olha daqui para saber onde alocar.
        """
    )

    # ---- Parâmetros ----
    with st.expander("⚙️ Parâmetros da análise (defaults validados pelo notebook)"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opp_window = st.selectbox(
                "Janela iFat (pregões)",
                [30, 60, 90, 120],
                index=2,  # 90 é o sweet spot do grid search
                key="opp_window",
            )
        with c2:
            opp_threshold = st.number_input(
                "Limiar iFat",
                value=0.65,
                min_value=0.50,
                max_value=0.80,
                step=0.01,
                key="opp_thr",
            )
        with c3:
            opp_horizon = st.selectbox(
                "Horizonte de avaliação (dias)",
                [30, 60, 90, 180, 365],
                index=2,  # 90d é onde o sinal costuma ter pico de efeito
                key="opp_horizon",
            )
        with c4:
            opp_min_events = st.number_input(
                "Mínimo de eventos para FORTE",
                value=50,
                min_value=10,
                max_value=500,
                step=10,
                key="opp_min_ev",
                help="Títulos com menos eventos são rotulados 'SEM DADOS' para evitar conclusão estatisticamente frágil.",
            )

    # ---- Computa ----
    with st.spinner("Calculando histórico + status atual de cada título..."):
        try:
            opp_df = compute_opportunities_table(
                df,
                window=int(opp_window),
                convention="mad_over_std",
                absolute_threshold=float(opp_threshold),
                main_horizon=int(opp_horizon),
                min_events_for_reliability=int(opp_min_events),
            )
        except Exception as e:
            st.error(f"Erro ao computar oportunidades: {e}")
            opp_df = pd.DataFrame()

    if opp_df.empty:
        st.warning(
            "Não foi possível montar a tabela. Verifique se o PU está disponível "
            "para os títulos da base selecionada."
        )
    else:
        # ---- Resumo de status ----
        total = len(opp_df)
        counts = opp_df["veredito"].value_counts().to_dict()

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("✅ FORTE", counts.get("✅ FORTE", 0))
        with col_s2:
            st.metric("🟢 OK", counts.get("🟢 OK", 0))
        with col_s3:
            st.metric("🟡 FRACO", counts.get("🟡 FRACO", 0))
        with col_s4:
            st.metric("❌ EVITAR", counts.get("❌ EVITAR", 0))
        with col_s5:
            st.metric("📊 SEM DADOS", counts.get("📊 SEM DADOS", 0))

        # ---- Filtros de exibição ----
        st.markdown("#### 🔍 Filtros")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            vereditos_opcoes = list(opp_df["veredito"].unique())
            vereditos_filtro = st.multiselect(
                "Mostrar vereditos",
                options=vereditos_opcoes,
                default=[v for v in vereditos_opcoes if "EVITAR" not in v and "SEM DADOS" not in v],
                key="opp_verf",
            )
        with col_f2:
            pos_min = st.selectbox(
                "Posição-alvo mínima",
                ["Qualquer", "≥ 25%", "≥ 50%", "≥ 75%", "100% (pânico)"],
                index=0,
                key="opp_pos_min",
            )
        with col_f3:
            ordenar_por = st.selectbox(
                "Ordenar por",
                [
                    "Veredito (melhor primeiro)",
                    f"Excesso {opp_horizon}d (maior)",
                    "Hit rate (maior)",
                    "Posição-alvo atual (maior)",
                    "Anos até vencimento (maior)",
                ],
                key="opp_order",
            )

        # Aplica filtros
        view = opp_df.copy()
        if vereditos_filtro:
            view = view[view["veredito"].isin(vereditos_filtro)]
        pos_map = {"Qualquer": 0, "≥ 25%": 25, "≥ 50%": 50, "≥ 75%": 75, "100% (pânico)": 100}
        view = view[view["posicao_alvo_%"] >= pos_map.get(pos_min, 0)]

        # Ordenação
        if ordenar_por.startswith("Excesso"):
            view = view.sort_values(f"excesso_{opp_horizon}d_%", ascending=False, na_position="last")
        elif ordenar_por.startswith("Hit"):
            view = view.sort_values("hit_rate_%", ascending=False, na_position="last")
        elif ordenar_por.startswith("Posição"):
            view = view.sort_values("posicao_alvo_%", ascending=False)
        elif ordenar_por.startswith("Anos"):
            view = view.sort_values("anos_ate_venc", ascending=False)
        # default: já vem ordenado por score

        # ---- Tabela principal ----
        st.markdown("#### 📋 Tabela de oportunidades")

        cols_show = [
            "titulo",
            "vencimento",
            "anos_ate_venc",
            "iFat_atual",
            "posicao_alvo_%",
            f"excesso_{opp_horizon}d_%",
            "hit_rate_%",
            "n_eventos_hist",
            "veredito",
            "recomendacao",
        ]
        st.dataframe(view[cols_show], width="stretch", hide_index=True)
        download_btn(view, "oportunidades.csv")

        # ---- Destaque: títulos em zona de entrada ATIVA ----
        ativos = opp_df[opp_df["posicao_alvo_%"] > 0].copy()
        ativos = ativos[~ativos["veredito"].isin(["❌ EVITAR", "📊 SEM DADOS"])]

        if not ativos.empty:
            st.markdown("---")
            st.markdown("### 🔥 Ação recomendada AGORA")
            st.caption(
                "Estes são os títulos onde o iFat está em zona de entrada **E** "
                "o histórico valida o sinal. Considere alocar conforme a posição-alvo."
            )

            for _, r in ativos.head(5).iterrows():
                emoji = "🚨" if r["posicao_alvo_%"] == 100 else ("⚡" if r["posicao_alvo_%"] >= 75 else "📍")
                excesso_col = f"excesso_{opp_horizon}d_%"
                excesso_val = r.get(excesso_col, 0) or 0
                hit_val = r.get("hit_rate_%", 0) or 0

                with st.container():
                    col_a, col_b, col_c = st.columns([3, 2, 2])
                    with col_a:
                        st.markdown(
                            f"**{emoji} {r['titulo']}** · vence {r['vencimento']} "
                            f"({r['anos_ate_venc']:.1f} anos)"
                        )
                        st.caption(r["nota"])
                    with col_b:
                        st.metric(
                            "Posição-alvo",
                            f"{r['posicao_alvo_%']:.0f}%",
                            delta=f"iFat {r['iFat_atual']:.3f}",
                        )
                    with col_c:
                        st.metric(
                            f"Excesso histórico {opp_horizon}d",
                            f"{excesso_val:+.2f}%",
                            delta=f"hit {hit_val:.0f}%",
                        )
                    st.markdown("---")
        else:
            st.info(
                "📭 Nenhum título em zona de entrada ativa no momento. "
                "O iFat de todos os títulos válidos está em regime normal — aguarde."
            )

        # ---- Estatísticas finais ----
        with st.expander("📊 Estatísticas do universo analisado"):
            st.markdown(f"""
            - **Total de títulos analisados**: {len(opp_df)}
            - **Com dados suficientes (≥10 eventos)**: {(opp_df["n_eventos_hist"] >= 10).sum()}
            - **Onde o sinal é forte ou OK**: {counts.get("✅ FORTE", 0) + counts.get("🟢 OK", 0)}
            - **Onde o sinal deve ser evitado**: {counts.get("❌ EVITAR", 0)}
            - **Em zona de entrada ativa (posição > 0)**: {(opp_df["posicao_alvo_%"] > 0).sum()}
            """)


# ===================== TAB 2: Backtest =====================
with tab2:
    st.subheader("🧪 Validação (Backtest) — reversão é verificável aqui")

    pairs = (
        df[["titulo", "vencimento"]]
        .drop_duplicates()
        .sort_values(["titulo", "vencimento"])
    )
    pairs["label"] = pairs.apply(
        lambda r: f"{r['titulo']} | {r['vencimento'].date()}", axis=1
    )
    sel = st.selectbox("Escolha um título", pairs["label"].tolist(), key="bt_sel")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        window_days = st.selectbox("Janela do sinal (dias)", [180, 365, 730], index=1)
    with col_b:
        z_min = st.slider("Z mínimo (sinal forte)", 0.0, 3.0, 1.0, 0.1)
    with col_c:
        horizons = st.multiselect(
            "Horizontes (dias)", [30, 60, 90, 180, 365], default=[30, 90, 180]
        )

    titulo_sel, venc_sel = sel.split(" | ")
    venc_sel = pd.to_datetime(venc_sel)

    bt = backtest_mean_reversion(
        df,
        titulo=titulo_sel,
        vencimento=venc_sel,
        window_days=int(window_days),
        min_points=80,
        j_trigger="J4",
        z_min=float(z_min),
        horizons=tuple(horizons) if horizons else (90,),
    )

    # ---- Métricas resumo em cards ----
    summary = bt["summary"]
    st.markdown(f"**Eventos disparados:** {summary['n_events']}")

    if summary["n_events"] > 0 and "mean_ret_pu_90d" in summary:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Retorno médio PU (90d)", f"{summary['mean_ret_pu_90d']*100:.2f}%")
        with m2:
            st.metric(
                "Mediana (90d)",
                f"{summary.get('median_ret_pu_90d', 0)*100:.2f}%",
            )
        with m3:
            st.metric("Hit rate PU↑ (90d)", f"{summary['hit_pu_up_90d']*100:.1f}%")
        with m4:
            sharpe = summary.get("sharpe_like_pu_90d", np.nan)
            st.metric(
                "Sharpe-like (90d)",
                f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a",
            )

    with st.expander("Resumo completo do backtest"):
        st.json(summary)

    events = bt["events"]
    st.write("Eventos (datas em que o sinal disparou):")
    st.dataframe(events.head(200), width="stretch", hide_index=True)
    download_btn(events, f"backtest_events_{titulo_sel.replace(' ','_')}.csv")

    series = bt["series"].copy()

    st.subheader("📉 Série histórica (Taxa e PU)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series["data"], series["taxa"])
    ax.set_title("Taxa ao longo do tempo")
    ax.set_xlabel("Data")
    ax.set_ylabel("Taxa")
    st.pyplot(fig, clear_figure=True)

    if "pu" in series.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series["data"], series["pu"])
        ax.set_title("PU ao longo do tempo")
        ax.set_xlabel("Data")
        ax.set_ylabel("PU")
        st.pyplot(fig, clear_figure=True)

    # Distribuição do retorno em cada horizonte
    for h in horizons:
        col_name = f"ret_pu_{h}d"
        if col_name in events.columns:
            vals = events[col_name].dropna().values
            if len(vals) > 0:
                st.subheader(f"📊 Distribuição do retorno do PU após o sinal ({h}d)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(vals, bins=25)
                ax.axvline(0, color="red", linestyle="--", alpha=0.7)
                ax.axvline(
                    np.mean(vals), color="green", linestyle="-", alpha=0.7, label="média"
                )
                ax.set_xlabel(f"Retorno do PU ({h}d)")
                ax.set_ylabel("Frequência")
                ax.legend()
                st.pyplot(fig, clear_figure=True)


# ===================== TAB 3: Risco (Duration) =====================
with tab3:
    st.subheader("📐 Risco — Duration, Convexidade e DV01")
    st.caption(
        "Métricas de risco de taxa de juros. "
        "**Modified Duration** = sensibilidade % do PU a +1% na taxa. "
        "**DV01** = quanto o PU muda (em R$) a cada 0,01% (1bp) de variação na taxa."
    )

    pairs = (
        df[["titulo", "vencimento"]]
        .drop_duplicates()
        .sort_values(["titulo", "vencimento"])
    )
    pairs["label"] = pairs.apply(
        lambda r: f"{r['titulo']} | {r['vencimento'].date()}", axis=1
    )

    view_mode = st.radio(
        "Modo", ["Individual", "Comparar vários títulos"], horizontal=True
    )

    if view_mode == "Individual":
        sel_r = st.selectbox("Título", pairs["label"].tolist(), key="risk_sel")
        titulo_r, venc_r = sel_r.split(" | ")
        venc_r = pd.to_datetime(venc_r)
        s_r = get_series(df, titulo_r, venc_r)

        if "pu" not in s_r.columns or s_r["pu"].isna().all():
            st.error("PU não disponível para este título com a coluna selecionada.")
        else:
            taxa_r = float(s_r["taxa"].iloc[-1])
            pu_r = float(s_r["pu"].iloc[-1])
            years_r = max(
                (venc_r.normalize() - pd.Timestamp.today().normalize()).days / 365.25,
                0.01,
            )

            m = compute_risk_metrics(titulo_r, taxa_r, pu_r, years_r)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Duration (anos)", f"{m['duration_macaulay_a']:.2f}")
            with c2:
                st.metric(
                    "Modified Duration", f"{m['duration_modified_a']:.2f}"
                )
            with c3:
                st.metric("Convexidade", f"{m['convexidade_a']:.2f}")
            with c4:
                st.metric("DV01 (R$)", f"R$ {m['dv01_R$']:.2f}")

            st.caption(
                f"Tipo de fluxo assumido: **{m['tipo_fluxo']}** · "
                f"Taxa: {m['taxa_a']*100:.3f}% a.a. · "
                f"PU: {fmt_brl(pu_r)} · "
                f"Prazo: {years_r:.2f} anos"
            )

            st.markdown("### 📊 Tabela de sensibilidade do PU")
            st.caption(
                "Aproximação de 2ª ordem: ΔPU/PU ≈ -ModDur × Δy + ½ × Conv × (Δy)². "
                "Permite ler de imediato o ganho/perda em reais para cada cenário de taxa."
            )
            sens = pu_sensitivity_table(pu_r, taxa_r, years_r, titulo_r)
            st.dataframe(sens, width="stretch", hide_index=True)
            download_btn(
                sens, f"sensibilidade_{titulo_r.replace(' ','_')}.csv"
            )

            # gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sens["delta_bps"], sens["variacao_%"], marker="o")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Variação da taxa (bps)")
            ax.set_ylabel("Variação do PU (%)")
            ax.set_title(f"Sensibilidade do PU – {titulo_r}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

    else:  # Comparar vários
        sel_multi = st.multiselect(
            "Selecione títulos para comparar",
            pairs["label"].tolist(),
            default=pairs["label"].tolist()[:5],
        )

        if sel_multi:
            rows = []
            for s in sel_multi:
                titulo_r, venc_r = s.split(" | ")
                venc_r = pd.to_datetime(venc_r)
                s_r = get_series(df, titulo_r, venc_r)
                if "pu" not in s_r.columns or s_r["pu"].isna().all():
                    continue
                taxa_r = float(s_r["taxa"].iloc[-1])
                pu_r = float(s_r["pu"].iloc[-1])
                years_r = max(
                    (
                        venc_r.normalize()
                        - pd.Timestamp.today().normalize()
                    ).days
                    / 365.25,
                    0.01,
                )
                m = compute_risk_metrics(titulo_r, taxa_r, pu_r, years_r)
                rows.append(
                    {
                        "titulo": titulo_r,
                        "vencimento": venc_r.date(),
                        "anos_ate_venc": round(years_r, 2),
                        "taxa_%a.a.": round(m["taxa_a"] * 100, 3),
                        "pu": round(pu_r, 2),
                        "duration": round(m["duration_macaulay_a"], 3),
                        "mod_duration": round(m["duration_modified_a"], 3),
                        "convexidade": round(m["convexidade_a"], 3),
                        "DV01_R$": round(m["dv01_R$"], 3),
                        "tipo_fluxo": m["tipo_fluxo"],
                    }
                )

            comp_df = pd.DataFrame(rows).sort_values("mod_duration", ascending=False)
            st.dataframe(comp_df, width="stretch", hide_index=True)
            download_btn(comp_df, "risco_comparativo.csv")

            if not comp_df.empty:
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.scatter(
                    comp_df["anos_ate_venc"],
                    comp_df["mod_duration"],
                    s=comp_df["DV01_R$"] * 30 + 40,
                    alpha=0.7,
                )
                for _, r in comp_df.iterrows():
                    ax.annotate(
                        f"{r['titulo'].split()[-1]} ({r['vencimento'].year})",
                        (r["anos_ate_venc"], r["mod_duration"]),
                        fontsize=8,
                        xytext=(5, 5),
                        textcoords="offset points",
                    )
                ax.set_xlabel("Anos até o vencimento")
                ax.set_ylabel("Modified Duration")
                ax.set_title("Duration vs Prazo (bolha = DV01 em R$)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)


# ===================== TAB 4: Curvas & Inflação =====================
with tab4:
    st.subheader("🌐 Curvas de juros & Inflação implícita")

    ref_dates = sorted(df["data"].unique(), reverse=True)
    ref_date = st.selectbox(
        "Data de referência (último snapshot disponível por padrão)",
        ref_dates[:30],
        index=0,
    )
    ref_date = pd.Timestamp(ref_date)

    st.markdown("### Curva de juros do Tesouro Direto")

    curve_pre = build_yield_curve(df, familia="pre", ref_date=ref_date)
    curve_ipca = build_yield_curve(df, familia="ipca", ref_date=ref_date)

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    if not curve_pre.empty:
        ax.plot(
            curve_pre["anos_ate_venc"],
            curve_pre["taxa_atual"],
            marker="o",
            label="Prefixado",
            color="#2E86AB",
        )
        plotted = True
    if not curve_ipca.empty:
        ax.plot(
            curve_ipca["anos_ate_venc"],
            curve_ipca["taxa_atual"],
            marker="s",
            label="IPCA+ (taxa real)",
            color="#A23B72",
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Anos até o vencimento")
        ax.set_ylabel("Taxa (% a.a.)")
        ax.set_title(f"Curva de juros – {pd.Timestamp(ref_date).date()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Sem títulos suficientes nessa data para construir a curva.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Curva prefixada**")
        if not curve_pre.empty:
            st.dataframe(curve_pre, width="stretch", hide_index=True)
        else:
            st.caption("(sem dados)")
    with col_c2:
        st.markdown("**Curva IPCA+ (taxa real)**")
        if not curve_ipca.empty:
            st.dataframe(curve_ipca, width="stretch", hide_index=True)
        else:
            st.caption("(sem dados)")

    st.markdown("---")
    st.markdown("### 💱 Inflação implícita (pré vs IPCA+)")
    st.caption(
        "Para cada par de vencimentos próximos, a inflação implícita é dada por "
        "**(1+pré)/(1+real) − 1**. "
        "Se você acredita que a inflação futura será **maior** que a implícita, "
        "IPCA+ tende a render mais que Prefixado para aquele prazo."
    )

    tol = st.slider(
        "Tolerância de diferença entre vencimentos (anos)", 0.25, 2.5, 1.0, 0.25
    )
    inf_df = implied_inflation(df, ref_date=ref_date, tol_years=tol)
    if not inf_df.empty:
        st.dataframe(inf_df, width="stretch", hide_index=True)
        download_btn(inf_df, "inflacao_implicita.csv")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            inf_df["prazo_medio_a"],
            inf_df["inflacao_implicita_%a.a."],
            marker="o",
            color="#F18F01",
        )
        ax.set_xlabel("Prazo médio (anos)")
        ax.set_ylabel("Inflação implícita (% a.a.)")
        ax.set_title("Curva de inflação implícita")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info(
            "Não foi possível formar pares pré × IPCA+ com essa tolerância. "
            "Aumente a tolerância ou escolha outra data."
        )

    st.markdown("---")
    with st.expander("🏛️ ETTJ ANBIMA (opcional — requer `pyettj`)"):
        st.caption(
            "Busca a ETTJ oficial ANBIMA (curva PRE/IPCA suavizada via Svensson) "
            "dos últimos dias úteis. Requer a biblioteca `pyettj` instalada. "
            "Se falhar, a curva TD acima já resolve a maioria dos casos."
        )
        if st.button("Buscar ETTJ ANBIMA"):
            col_an1, col_an2 = st.columns(2)
            with col_an1:
                st.markdown("**Pré**")
                res_pre = fetch_anbima_ettj("PRE")
                if res_pre is not None:
                    st.dataframe(res_pre.head(30), width="stretch")
                else:
                    st.warning("Indisponível (biblioteca/rede/parsing).")
            with col_an2:
                st.markdown("**IPCA**")
                res_ipca = fetch_anbima_ettj("IPCA")
                if res_ipca is not None:
                    st.dataframe(res_ipca.head(30), width="stretch")
                else:
                    st.warning("Indisponível (biblioteca/rede/parsing).")


# ===================== TAB 5: Cenários =====================
with tab5:
    st.subheader("📈 Cenários (R$) — PU futuro, carrego e venda antecipada")

    pairs = (
        df[["titulo", "vencimento"]]
        .drop_duplicates()
        .sort_values(["titulo", "vencimento"])
    )
    pairs["label"] = pairs.apply(
        lambda r: f"{r['titulo']} | {r['vencimento'].date()}", axis=1
    )
    sel2 = st.selectbox("Título para cenários", pairs["label"].tolist(), key="sel2")

    titulo2, venc2 = sel2.split(" | ")
    venc2 = pd.to_datetime(venc2)
    s = get_series(df, titulo2, venc2)

    if "pu" not in s.columns or s["pu"].isna().all():
        st.error("PU não disponível para este título com a coluna selecionada.")
        st.stop()

    taxa_atual = float(s["taxa"].iloc[-1])
    pu_atual = float(s["pu"].iloc[-1])
    fam = classify_family(titulo2)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Taxa atual (%)", f"{taxa_atual:.3f}")
    with m2:
        st.metric("PU atual", fmt_brl(pu_atual))
    with m3:
        st.metric("Família", fam.upper())

    # ---------- Sub-tabs dentro do Cenários ----------
    sc1, sc2, sc3 = st.tabs(
        ["PU futuro (MtM)", "Carrego até o vencimento", "Carrego vs Venda antecipada"]
    )

    # ==== Sub-tab 1: PU via regressão ====
    with sc1:
        st.caption(
            "Compra ao PU atual e projeta o PU futuro via regressão histórica PU~taxa. "
            "Use para simular variações de taxa (stress ou queda) e o impacto em R$."
        )

        lookback = st.selectbox(
            "Janela para calibrar PU~taxa (dias)",
            [180, 365, 730],
            index=2,
            key="lb_mtm",
        )
        aporte_mtm = st.number_input(
            "Aporte (R$)", min_value=0.0, value=100000.0, step=1000.0, key="aporte_mtm"
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            taxa_future_stress = st.number_input(
                "Taxa futura (stress: alta)",
                value=float(taxa_atual + 0.5),
                step=0.05,
                key="tf_stress",
            )
        with col_s2:
            taxa_future_base = st.number_input(
                "Taxa futura (base)",
                value=float(taxa_atual),
                step=0.05,
                key="tf_base",
            )
        with col_s3:
            taxa_future_otim = st.number_input(
                "Taxa futura (otimista: queda)",
                value=float(taxa_atual - 0.5),
                step=0.05,
                key="tf_otim",
            )

        try:
            est_base = estimate_pu_from_taxa(s, taxa_future_base, int(lookback))
            est_otim = estimate_pu_from_taxa(s, taxa_future_otim, int(lookback))
            est_stress = estimate_pu_from_taxa(s, taxa_future_stress, int(lookback))
        except ValueError as e:
            st.error(f"Não foi possível calibrar PU~taxa: {e}")
            st.stop()

        units = aporte_mtm / pu_atual if pu_atual > 0 else 0

        def sv(pu_est):
            val = units * pu_est
            ret = (val / aporte_mtm) - 1.0 if aporte_mtm > 0 else np.nan
            return val, ret

        v_b, r_b = sv(est_base["pu_est"])
        v_o, r_o = sv(est_otim["pu_est"])
        v_s, r_s = sv(est_stress["pu_est"])

        res = pd.DataFrame(
            [
                {
                    "cenario": "Stress (taxa sobe)",
                    "taxa_future": taxa_future_stress,
                    "pu_est": est_stress["pu_est"],
                    "valor_R$": v_s,
                    "retorno_%": 100 * r_s,
                },
                {
                    "cenario": "Base (taxa igual)",
                    "taxa_future": taxa_future_base,
                    "pu_est": est_base["pu_est"],
                    "valor_R$": v_b,
                    "retorno_%": 100 * r_b,
                },
                {
                    "cenario": "Otimista (taxa cai)",
                    "taxa_future": taxa_future_otim,
                    "pu_est": est_otim["pu_est"],
                    "valor_R$": v_o,
                    "retorno_%": 100 * r_o,
                },
            ]
        )
        st.dataframe(res, width="stretch", hide_index=True)
        download_btn(res, "cenarios_mtm.csv")

        st.markdown("#### Calibração PU ~ Taxa")
        end = s["data"].max()
        start_lb = end - pd.Timedelta(days=int(lookback))
        w = s[(s["data"] >= start_lb) & (s["data"] <= end)].dropna(
            subset=["taxa", "pu"]
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(w["taxa"], w["pu"], s=15, alpha=0.6)
        xline = np.linspace(w["taxa"].min(), w["taxa"].max(), 100)
        a, b = est_base["a"], est_base["b"]
        ax.plot(xline, a + b * xline, color="red")
        ax.set_xlabel("Taxa")
        ax.set_ylabel("PU")
        ax.set_title("PU vs Taxa (com regressão linear)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    # ==== Sub-tab 2: Carrego com DCA ====
    with sc2:
        st.caption(
            "Simula o rendimento até o vencimento assumindo IPCA estimado. "
            "Agora com suporte a **aportes periódicos (DCA)**, custódia e IR regressivo."
        )

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            aporte_carry = st.number_input(
                "Aporte inicial (R$)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                key="aporte_carry",
            )
        with col_k2:
            periodic_contrib = st.number_input(
                "Aporte periódico (R$) — 0 desliga",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key="periodic_contrib",
                help="Aporte extra adicionado em cada período (ex: mensal).",
            )

        contrib_freq = st.radio(
            "Frequência do aporte periódico",
            ["Mensal", "Anual"],
            horizontal=True,
            key="contrib_freq",
        )
        contrib_freq_code = "ME" if contrib_freq == "Mensal" else "YE"

        # Taxa real
        taxa_real_a = taxa_atual / 100.0 if taxa_atual > 1 else taxa_atual
        st.write({"taxa_real_%a.a.": round(taxa_real_a * 100, 3)})

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            ipca_baixo_a = (
                st.number_input(
                    "IPCA baixo (% a.a.)", value=3.0, step=0.25, key="ipca_baixo"
                )
                / 100.0
            )
        with col_i2:
            ipca_base_a = (
                st.number_input(
                    "IPCA base (% a.a.)", value=4.0, step=0.25, key="ipca_base"
                )
                / 100.0
            )
        with col_i3:
            ipca_alto_a = (
                st.number_input(
                    "IPCA alto (% a.a.)", value=6.0, step=0.25, key="ipca_alto"
                )
                / 100.0
            )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            include_fees = st.checkbox(
                "Incluir custódia (aprox.)", value=True, key="fees_carry"
            )
            custody_fee_a = (
                st.number_input(
                    "Custódia (% a.a.)",
                    value=0.20,
                    step=0.05,
                    key="custody_fee",
                )
                / 100.0
            )
        with col_f2:
            apply_ir = st.checkbox(
                "IR regressivo automático",
                value=True,
                key="apply_ir",
            )

        start_date = pd.Timestamp.today().normalize()
        maturity_date = venc2.normalize()

        common_kwargs = dict(
            aporte=aporte_carry,
            real_rate_a=taxa_real_a,
            start_date=start_date,
            maturity_date=maturity_date,
            freq="ME",
            include_fees=include_fees,
            custody_fee_a=custody_fee_a,
            apply_ir=apply_ir,
            periodic_contribution=periodic_contrib,
            contribution_freq=contrib_freq_code,
        )

        df_c_base = simulate_carry_ipca(ipca_rate_a=ipca_base_a, **common_kwargs)
        df_c_baixo = simulate_carry_ipca(ipca_rate_a=ipca_baixo_a, **common_kwargs)
        df_c_alto = simulate_carry_ipca(ipca_rate_a=ipca_alto_a, **common_kwargs)

        def final_value(df_):
            if "valor_liquido_final_ir" in df_.columns:
                return float(df_["valor_liquido_final_ir"].iloc[-1])
            return float(df_["valor_pos_fees"].iloc[-1])

        total_aportado = float(df_c_base["aportado_acumulado"].iloc[-1])

        res_carry = pd.DataFrame(
            [
                {
                    "cenário": "IPCA baixo",
                    "IPCA_%a.a.": round(ipca_baixo_a * 100, 2),
                    "total_aportado": total_aportado,
                    "valor_final_R$": final_value(df_c_baixo),
                    "ganho_liq_R$": final_value(df_c_baixo) - total_aportado,
                    "retorno_total_%": 100
                    * (final_value(df_c_baixo) / total_aportado - 1)
                    if total_aportado > 0
                    else 0,
                },
                {
                    "cenário": "IPCA base",
                    "IPCA_%a.a.": round(ipca_base_a * 100, 2),
                    "total_aportado": total_aportado,
                    "valor_final_R$": final_value(df_c_base),
                    "ganho_liq_R$": final_value(df_c_base) - total_aportado,
                    "retorno_total_%": 100
                    * (final_value(df_c_base) / total_aportado - 1)
                    if total_aportado > 0
                    else 0,
                },
                {
                    "cenário": "IPCA alto",
                    "IPCA_%a.a.": round(ipca_alto_a * 100, 2),
                    "total_aportado": total_aportado,
                    "valor_final_R$": final_value(df_c_alto),
                    "ganho_liq_R$": final_value(df_c_alto) - total_aportado,
                    "retorno_total_%": 100
                    * (final_value(df_c_alto) / total_aportado - 1)
                    if total_aportado > 0
                    else 0,
                },
            ]
        )
        st.dataframe(res_carry, width="stretch", hide_index=True)
        download_btn(res_carry, "carrego_resumo.csv")

        if apply_ir and "aliquota_ir" in df_c_base.columns:
            st.caption(
                f"IR no vencimento: "
                f"{df_c_base['aliquota_ir'].iloc[-1]*100:.1f}% (regressivo por prazo)"
            )

        ycol = (
            "valor_liquido_final_ir"
            if ("valor_liquido_final_ir" in df_c_base.columns and apply_ir)
            else "valor_pos_fees"
        )
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(
            df_c_baixo["data"], df_c_baixo[ycol], label="IPCA baixo", alpha=0.85
        )
        ax2.plot(df_c_base["data"], df_c_base[ycol], label="IPCA base", linewidth=2)
        ax2.plot(df_c_alto["data"], df_c_alto[ycol], label="IPCA alto", alpha=0.85)
        if periodic_contrib > 0:
            ax2.plot(
                df_c_base["data"],
                df_c_base["aportado_acumulado"],
                label="Total aportado",
                linestyle="--",
                color="gray",
            )
        ax2.set_title("Carrego (valor estimado ao longo do tempo)")
        ax2.set_xlabel("Data")
        ax2.set_ylabel("R$")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

    # ==== Sub-tab 3: Carrego vs Venda antecipada ====
    with sc3:
        st.caption(
            "Compara: segurar até o vencimento **vs** vender em N anos assumindo taxa futura. "
            "O PU futuro é estimado via regressão histórica PU~taxa. "
            "(Disponível para IPCA+ — usa taxa real + IPCA assumido.)"
        )

        if fam != "ipca":
            st.info(
                "Esta simulação está otimizada para IPCA+. "
                "Para outras famílias, use a sub-aba 'PU futuro (MtM)'."
            )
        else:
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                aporte_v = st.number_input(
                    "Aporte (R$)",
                    min_value=0.0,
                    value=100000.0,
                    step=1000.0,
                    key="aporte_v",
                )
            with col_v2:
                anos_max = max(
                    0.5,
                    (
                        venc2.normalize() - pd.Timestamp.today().normalize()
                    ).days
                    / 365.25
                    - 0.1,
                )
                sell_years = st.slider(
                    "Vender em quantos anos?",
                    0.5,
                    float(round(anos_max, 1)),
                    min(2.0, float(round(anos_max, 1))),
                    0.5,
                )
            with col_v3:
                ipca_assumido = (
                    st.number_input(
                        "IPCA assumido no período (% a.a.)",
                        value=4.0,
                        step=0.25,
                        key="ipca_assumido",
                    )
                    / 100.0
                )

            col_v4, col_v5 = st.columns(2)
            with col_v4:
                taxa_future_venda = st.number_input(
                    "Taxa esperada na venda (% a.a.)",
                    value=float(taxa_atual),
                    step=0.05,
                    key="taxa_venda",
                )
            with col_v5:
                lookback_v = st.selectbox(
                    "Janela PU~taxa", [180, 365, 730], index=2, key="lb_v"
                )

            try:
                comp = simulate_sell_before_maturity(
                    s,
                    titulo2,
                    aporte=aporte_v,
                    sell_years=float(sell_years),
                    taxa_future=float(taxa_future_venda),
                    ipca_rate_a=float(ipca_assumido),
                    lookback_days=int(lookback_v),
                )
            except ValueError as e:
                st.error(f"Erro: {e}")
                st.stop()

            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric(
                    "Venda antecipada (líquido)",
                    fmt_brl(comp["valor_venda_liquido"]),
                )
            with d2:
                st.metric(
                    f"Carrego {sell_years:.1f}a (líquido)",
                    fmt_brl(comp["valor_carry_liquido"]),
                )
            with d3:
                diff = comp["diferenca_liquida"]
                st.metric(
                    "Diferença (venda - carrego)",
                    fmt_brl(diff),
                    delta=f"{'venda ganha' if diff > 0 else 'carrego ganha'}",
                )

            comp_df = pd.DataFrame(
                [
                    {
                        "métrica": k,
                        # valor mistura floats (R$) e strings (titulo/vantagem);
                        # uniformiza para texto p/ não quebrar a serialização Arrow.
                        "valor": f"{v:,.2f}" if isinstance(v, (int, float)) else str(v),
                    }
                    for k, v in comp.items()
                ]
            )
            with st.expander("Detalhes do cálculo"):
                st.dataframe(comp_df, width="stretch", hide_index=True)
                download_btn(comp_df, "carrego_vs_venda.csv")

            st.caption(
                f"💡 **Leitura**: se a taxa está em **{taxa_atual:.2f}%** hoje e cair para "
                f"**{taxa_future_venda:.2f}%** em {sell_years:.1f} anos, "
                f"a **{comp['vantagem']}** é a estratégia melhor neste cenário. "
                "Faça stress com taxa mais alta para ver o risco do MtM."
            )


# ===================== TAB 6: Fat Tails =====================
with tab6:
    st.subheader("🐘 Índice de Fat Tail (Taleb) — renda fixa")

    st.markdown(
        """
        **O que é isso?** A razão entre **MAD** (desvio absoluto médio) e **STD** (desvio-padrão)
        detecta **caudas gordas** nos retornos.

        - **Distribuição Normal (gaussiana)**: MAD/STD ≈ **0,7979** (= √(2/π))
        - **iFat < 0,7979** → caudas **mais gordas** que o normal (stress, pânico)
        - **iFat > 0,7979** → caudas **mais finas** que o normal (mercado calmo)

        Em renda fixa de longa duração (IPCA+ longos), quando o iFat cai muito abaixo da média
        histórica do próprio título, historicamente isso coincide com **janelas de entrada**:
        o PU está em pânico e tende a reverter.

        Referência: Taleb, *Statistical Consequences of Fat Tails*, cap. 4.4.1.
        """
    )

    pairs_ft = (
        df[["titulo", "vencimento"]]
        .drop_duplicates()
        .sort_values(["titulo", "vencimento"])
    )
    pairs_ft["label"] = pairs_ft.apply(
        lambda r: f"{r['titulo']} | {r['vencimento'].date()}", axis=1
    )

    sel_ft = st.selectbox(
        "Escolha o título", pairs_ft["label"].tolist(), key="ft_sel_tab"
    )
    titulo_ft, venc_ft = sel_ft.split(" | ")
    venc_ft = pd.to_datetime(venc_ft)
    s_ft = get_series(df, titulo_ft, venc_ft)

    if "pu" not in s_ft.columns or s_ft["pu"].isna().all():
        st.error("PU não disponível para este título com a coluna selecionada.")
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            window_ft = st.selectbox(
                "Janela móvel (pregões)",
                [30, 60, 90, 120, 180],
                index=1,
                key="ft_window_tab",
            )
        with col_cfg2:
            conv_label = st.selectbox(
                "Convenção",
                ["MAD/STD (notebook original)", "STD/MAD (Taleb)"],
                index=0,
                key="ft_conv",
            )
            convention = (
                "mad_over_std"
                if "MAD/STD" in conv_label
                else "std_over_mad"
            )
        with col_cfg3:
            abs_thr = st.number_input(
                "Limiar absoluto",
                value=0.65 if convention == "mad_over_std" else 1.45,
                step=0.05,
                key="ft_thr_tab",
                help=(
                    "MAD/STD: limiar típico 0,65 (bem abaixo do gaussiano 0,798). "
                    "STD/MAD: limiar típico 1,45 (acima do gaussiano 1,253)."
                ),
            )

        ft_df = compute_fat_tail_index(
            s_ft, window=int(window_ft), price_col="pu", convention=convention
        )
        ft_df = fat_tail_entry_signals(
            ft_df, absolute_threshold=float(abs_thr), use_moving_band=True
        )

        if ft_df.empty:
            st.warning(
                f"Não há dados suficientes ({window_ft} pregões) para este título."
            )
        else:
            # ---------- Métricas resumo ----------
            summ = fat_tail_summary(ft_df)
            gaussian_ref = summ["gaussian_ref"]
            cur = summ["valor_atual"]

            if convention == "mad_over_std":
                status = (
                    "🔴 Cauda gorda"
                    if cur < gaussian_ref
                    else "🟢 Cauda fina"
                )
                distance = (gaussian_ref - cur) / gaussian_ref * 100
                distance_label = f"{distance:+.1f}% vs gaussiano"
            else:
                status = (
                    "🔴 Cauda gorda"
                    if cur > gaussian_ref
                    else "🟢 Cauda fina"
                )
                distance = (cur - gaussian_ref) / gaussian_ref * 100
                distance_label = f"{distance:+.1f}% vs gaussiano"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("iFat atual", f"{cur:.4f}", delta=distance_label)
            with c2:
                st.metric("Referência gaussiana", f"{gaussian_ref:.4f}")
            with c3:
                st.metric("Status", status)
            with c4:
                st.metric(
                    "% tempo em cauda gorda",
                    f"{summ['pct_em_cauda_gorda']*100:.1f}%",
                )

            with st.expander("📊 Estatísticas completas do iFat"):
                st.json(summ)

            # ---------- Gráfico duplo: PU + iFat ----------
            st.markdown("### 📈 PU & iFat ao longo do tempo")
            st.caption(
                "Marcamos em vermelho as **zonas de entrada** (iFat abaixo/acima da banda móvel "
                "E além do limiar absoluto). Nessas datas, historicamente o mercado tende a "
                "estar em pânico."
            )

            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

            # PU em cima, destacando zonas de entrada em vermelho
            mask_entry = ft_df["entrada"].values
            ax0 = axes[0]
            ax0.plot(
                ft_df["data"], ft_df["pu"], color="#2E86AB", linewidth=1.2, label="PU"
            )
            pu_entry = ft_df["pu"].where(ft_df["entrada"], np.nan)
            ax0.scatter(
                ft_df["data"][mask_entry],
                pu_entry[mask_entry],
                color="red",
                s=25,
                zorder=3,
                label="Zona de entrada",
            )
            ax0.set_ylabel("PU (R$)")
            ax0.set_title(f"PU — {titulo_ft}")
            ax0.legend(loc="best")
            ax0.grid(True, alpha=0.3)

            # iFat embaixo
            ax1 = axes[1]
            ax1.plot(
                ft_df["data"],
                ft_df["iFat"],
                color="#6A4C93",
                linewidth=1.0,
                label="iFat",
            )
            ax1.plot(
                ft_df["data"],
                ft_df["iFat_ma"],
                color="#F77F00",
                linewidth=1.0,
                linestyle="--",
                label="Média móvel",
            )
            ax1.plot(
                ft_df["data"],
                ft_df["iFat_ma_minus_sd"],
                color="#D62828",
                linewidth=0.9,
                linestyle=":",
                label="Banda de entrada",
            )
            ax1.axhline(
                gaussian_ref,
                color="black",
                linewidth=1.0,
                linestyle="-",
                alpha=0.5,
                label=f"Gaussiano = {gaussian_ref:.4f}",
            )
            ax1.axhline(
                abs_thr,
                color="gray",
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
                label=f"Limiar = {abs_thr}",
            )
            ax1.set_ylabel("iFat")
            ax1.set_xlabel("Data")
            ax1.set_title(f"Índice Fat Tail — janela {window_ft} pregões ({convention})")
            ax1.legend(loc="best", fontsize=9)
            ax1.grid(True, alpha=0.3)

            st.pyplot(fig, clear_figure=True)

            # ---------- Histograma ----------
            st.markdown("### 📊 Distribuição histórica do iFat")
            fig_hist, ax_h = plt.subplots(figsize=(11, 4))
            ax_h.hist(ft_df["iFat"].dropna(), bins=40, alpha=0.75, color="#6A4C93")
            ax_h.axvline(
                gaussian_ref,
                color="black",
                linestyle="-",
                linewidth=1.5,
                label=f"Gaussiano = {gaussian_ref:.4f}",
            )
            ax_h.axvline(
                cur,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Atual = {cur:.4f}",
            )
            ax_h.axvline(
                abs_thr,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label=f"Limiar = {abs_thr}",
            )
            ax_h.set_xlabel("iFat")
            ax_h.set_ylabel("Frequência")
            ax_h.legend()
            ax_h.grid(True, alpha=0.3)
            st.pyplot(fig_hist, clear_figure=True)

            st.caption(
                "💡 A distribuição costuma ser **bimodal**: uma moda perto do valor "
                "gaussiano (mercado normal) e outra abaixo dele (regime de stress). "
                "Os momentos mais lucrativos de compra tendem a estar na moda inferior."
            )

            # ---------- Backtest da estratégia ----------
            st.markdown("---")
            st.markdown("### 🧪 Backtest da estratégia Fat Tail")
            st.caption(
                "Quando o sinal dispara, qual é o retorno médio do PU no futuro? "
                "O **excesso vs baseline** mostra se a estratégia entrega retorno superior à média."
            )

            horizons_ft = st.multiselect(
                "Horizontes (dias)",
                [30, 60, 90, 180, 365],
                default=[30, 60, 90, 180],
                key="ft_horizons",
            )

            if horizons_ft:
                bt_ft = backtest_fat_tail_strategy(
                    s_ft,
                    window=int(window_ft),
                    price_col="pu",
                    convention=convention,
                    absolute_threshold=float(abs_thr),
                    use_moving_band=True,
                    horizons=tuple(horizons_ft),
                )

                s_bt = bt_ft["summary"]
                st.markdown(
                    f"**Eventos:** {s_bt['n_events']} "
                    f"(de {s_bt['total_obs']} observações → "
                    f"{s_bt['pct_em_entrada']*100:.1f}% do tempo)"
                )

                if s_bt["n_events"] > 0:
                    rows = []
                    for h in horizons_ft:
                        rows.append(
                            {
                                "horizonte": f"{h}d",
                                "retorno_medio_%": round(
                                    s_bt.get(f"mean_ret_{h}d", np.nan) * 100, 2
                                )
                                if s_bt.get(f"mean_ret_{h}d") is not None
                                else None,
                                "mediana_%": round(
                                    s_bt.get(f"median_ret_{h}d", np.nan) * 100, 2
                                )
                                if s_bt.get(f"median_ret_{h}d") is not None
                                else None,
                                "hit_rate_%": round(
                                    s_bt.get(f"hit_rate_{h}d", np.nan) * 100, 1
                                )
                                if s_bt.get(f"hit_rate_{h}d") is not None
                                else None,
                                "baseline_%": round(
                                    s_bt.get(f"baseline_mean_{h}d", np.nan) * 100, 2
                                )
                                if s_bt.get(f"baseline_mean_{h}d") is not None
                                else None,
                                "excesso_vs_base_%": round(
                                    s_bt.get(f"excesso_vs_baseline_{h}d", np.nan)
                                    * 100,
                                    2,
                                )
                                if s_bt.get(f"excesso_vs_baseline_{h}d") is not None
                                else None,
                                "sharpe_like": round(
                                    s_bt.get(f"sharpe_like_{h}d", np.nan), 2
                                )
                                if s_bt.get(f"sharpe_like_{h}d") is not None
                                else None,
                            }
                        )
                    bt_table = pd.DataFrame(rows)
                    st.dataframe(
                        bt_table, width="stretch", hide_index=True
                    )
                    download_btn(bt_table, f"fat_tail_bt_{titulo_ft.replace(' ','_')}.csv")

                    # Interpretação
                    best_horizon = max(
                        horizons_ft,
                        key=lambda h: s_bt.get(f"excesso_vs_baseline_{h}d", -999),
                    )
                    best_excess = s_bt.get(f"excesso_vs_baseline_{best_horizon}d", 0)
                    if best_excess > 0.002:  # >0,2% de excesso
                        st.success(
                            f"✅ A estratégia mostra excesso de retorno em {best_horizon}d "
                            f"(+{best_excess*100:.2f}% vs baseline). "
                            "Evidência (embora não conclusiva) a favor da tese de reversão."
                        )
                    elif best_excess < -0.002:
                        st.warning(
                            f"⚠️ Nos dados disponíveis, a estratégia teve desempenho pior que a média "
                            f"({best_excess*100:.2f}% vs baseline em {best_horizon}d). "
                            "Tese não suportada para este título."
                        )
                    else:
                        st.info(
                            "ℹ️ Resultado praticamente neutro vs baseline. "
                            "Talvez ajustar a janela ou o limiar."
                        )
                else:
                    st.info(
                        "Nenhum evento disparou com os parâmetros atuais. "
                        "Relaxe o limiar ou mude a janela."
                    )

            # ---------- Download dos dados brutos ----------
            with st.expander("⬇️ Baixar dados do iFat (CSV)"):
                download_btn(
                    ft_df[
                        [
                            "data",
                            "pu",
                            "retorno",
                            "MAD",
                            "STD",
                            "iFat",
                            "iFat_ma",
                            "iFat_ma_minus_sd",
                            "is_fat_tail",
                            "entrada",
                        ]
                    ],
                    f"ifat_{titulo_ft.replace(' ','_')}.csv",
                    label="⬇️ Baixar série histórica do iFat",
                )

            # ==========================================================
            # POSITION SIZING ESCALONADO
            # ==========================================================
            st.markdown("---")
            st.markdown("## 📊 Position Sizing Escalonado (entrada proporcional)")
            st.markdown(
                """
                **Problema que observamos:** o sinal binário tradicional (compra tudo ou nada)
                tende a **atrasar a entrada** porque depende da média móvel. Quando a banda
                finalmente desce, o PU já reagiu.

                **Solução:** em vez de esperar o "gatilho perfeito", construa a posição
                **escalonadamente** à medida que o iFat cai. Isso:

                - Melhora o **preço médio** (você compra na descida, não só no fundo)
                - **Protege contra falsos sinais** (se iFat tocar 0.72 e voltar, você fica com só 25% investido)
                - Entrega um **plano executável** sem precisar acertar o fundo exato
                """
            )

            st.markdown("### ⚙️ Configurar níveis de entrada")
            st.caption(
                "Cada linha diz: 'se o iFat está ABAIXO deste valor (e acima do próximo), a posição-alvo é X%'. "
                "Valores abaixo de 0.7979 (gaussiano) indicam stress."
            )

            # Usa session state para manter níveis customizados
            default_levels = DEFAULT_SIZING_LEVELS_MAD

            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
            with col_l1:
                thr_1 = st.number_input(
                    "Limiar 1 (inicia posição)",
                    value=0.75,
                    min_value=0.40,
                    max_value=0.85,
                    step=0.01,
                    key="sizing_thr1",
                    help="iFat abaixo disso → começa a entrar",
                )
                pos_1 = st.number_input(
                    "Posição (%)",
                    value=25,
                    min_value=0,
                    max_value=100,
                    step=5,
                    key="sizing_pos1",
                )
            with col_l2:
                thr_2 = st.number_input(
                    "Limiar 2",
                    value=0.70,
                    min_value=0.35,
                    max_value=0.80,
                    step=0.01,
                    key="sizing_thr2",
                )
                pos_2 = st.number_input(
                    "Posição (%) ",
                    value=50,
                    min_value=0,
                    max_value=100,
                    step=5,
                    key="sizing_pos2",
                )
            with col_l3:
                thr_3 = st.number_input(
                    "Limiar 3",
                    value=0.65,
                    min_value=0.30,
                    max_value=0.75,
                    step=0.01,
                    key="sizing_thr3",
                )
                pos_3 = st.number_input(
                    "Posição (%)  ",
                    value=75,
                    min_value=0,
                    max_value=100,
                    step=5,
                    key="sizing_pos3",
                )
            with col_l4:
                thr_4 = st.number_input(
                    "Limiar 4 (posição cheia)",
                    value=0.60,
                    min_value=0.25,
                    max_value=0.70,
                    step=0.01,
                    key="sizing_thr4",
                )
                pos_4 = st.number_input(
                    "Posição (%)   ",
                    value=100,
                    min_value=0,
                    max_value=100,
                    step=5,
                    key="sizing_pos4",
                )

            # Monta os níveis customizados (ordenados do maior threshold p/ menor)
            custom_levels = [
                (thr_1, 0.00),
                (thr_2, pos_1 / 100.0),
                (thr_3, pos_2 / 100.0),
                (thr_4, pos_3 / 100.0),
                (0.0, pos_4 / 100.0),
            ]

            # Mostra tabela de interpretação
            niveis_df = pd.DataFrame(
                [
                    {"faixa": f"iFat ≥ {thr_1:.2f}", "posição": "0%", "interpretação": "mercado normal"},
                    {"faixa": f"{thr_2:.2f} ≤ iFat < {thr_1:.2f}", "posição": f"{pos_1}%", "interpretação": "leve pressão"},
                    {"faixa": f"{thr_3:.2f} ≤ iFat < {thr_2:.2f}", "posição": f"{pos_2}%", "interpretação": "stress moderado"},
                    {"faixa": f"{thr_4:.2f} ≤ iFat < {thr_3:.2f}", "posição": f"{pos_3}%", "interpretação": "stress forte"},
                    {"faixa": f"iFat < {thr_4:.2f}", "posição": f"{pos_4}%", "interpretação": "pânico → posição cheia"},
                ]
            )
            st.dataframe(niveis_df, width="stretch", hide_index=True)

            # ---------- Estado atual ----------
            capital_total = st.number_input(
                "Capital total disponível (R$)",
                min_value=1000.0,
                value=100000.0,
                step=1000.0,
                key="capital_sizing",
            )

            with st.spinner("Calculando posição escalonada e rodando backtest..."):
                bt_sizing = backtest_fat_tail_sizing(
                    s_ft,
                    window=int(window_ft),
                    price_col="pu",
                    convention=convention,
                    levels=custom_levels,
                    capital_total=float(capital_total),
                    hold_horizons=(30, 60, 90, 180, 365),
                )

            sz = bt_sizing["summary"]

            # Decisão atual
            st.markdown("### 🎯 Decisão AGORA")
            pos_atual_pct = sz["posicao_atual_target_%"]
            valor_alocado = capital_total * pos_atual_pct / 100.0

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("iFat atual", f"{sz['iFat_atual']:.4f}")
            with col_d2:
                st.metric(
                    "Posição-alvo",
                    f"{pos_atual_pct:.0f}%",
                    delta=f"{valor_alocado:,.0f} R$" if valor_alocado > 0 else None,
                )
            with col_d3:
                caixa_pct = 100 - pos_atual_pct
                st.metric(
                    "Manter em caixa",
                    f"{caixa_pct:.0f}%",
                    delta=f"{capital_total - valor_alocado:,.0f} R$",
                )

            if pos_atual_pct == 0:
                st.info(
                    "🟢 **Mercado em regime normal** — sem ação recomendada. "
                    "Aguarde o iFat cair para os níveis de entrada."
                )
            elif pos_atual_pct < 50:
                st.warning(
                    f"🟡 **Pressão leve** — construa {pos_atual_pct:.0f}% da posição. "
                    "Se o stress continuar, o app vai sinalizar aumento."
                )
            elif pos_atual_pct < 100:
                st.error(
                    f"🟠 **Stress significativo** — posição-alvo {pos_atual_pct:.0f}%. "
                    "Momento de aumentar exposição de forma disciplinada."
                )
            else:
                st.error(
                    f"🔴 **PÂNICO DETECTADO** — posição cheia recomendada. "
                    "Historicamente esses são os melhores pontos de entrada em renda fixa longa."
                )

            # ---------- Gráfico: PU + Posição ao longo do tempo ----------
            st.markdown("### 📈 Evolução da posição ao longo do tempo")
            st.caption(
                "Acima: PU com cor-escala indicando a posição-alvo naquele momento. "
                "Abaixo: iFat + posição construída (em escala 0-100%)."
            )

            series_ft = bt_sizing["series_ft"]

            fig_sz, axes_sz = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

            # PU no topo
            ax0 = axes_sz[0]
            ax0.plot(series_ft["data"], series_ft["pu"], color="#2E86AB", linewidth=1.2, label="PU")
            # marca compras (pontos verdes, tamanho proporcional à compra)
            compras_evt = series_ft[series_ft["posicao_change"] > 0.001]
            if not compras_evt.empty:
                ax0.scatter(
                    compras_evt["data"],
                    compras_evt["pu"],
                    s=compras_evt["posicao_change"] * 300,
                    color="green",
                    alpha=0.7,
                    zorder=3,
                    label="Compras (tamanho ~ qtd)",
                )
            # marca vendas (pontos vermelhos)
            vendas_evt = series_ft[series_ft["posicao_change"] < -0.001]
            if not vendas_evt.empty:
                ax0.scatter(
                    vendas_evt["data"],
                    vendas_evt["pu"],
                    s=-vendas_evt["posicao_change"] * 300,
                    color="red",
                    alpha=0.6,
                    zorder=3,
                    label="Reduções",
                )
            ax0.set_ylabel("PU (R$)")
            ax0.set_title(f"PU — compras (verde) e reduções (vermelho) com tamanho proporcional")
            ax0.legend(loc="best")
            ax0.grid(True, alpha=0.3)

            # iFat + linhas de threshold
            ax1 = axes_sz[1]
            ax1.plot(series_ft["data"], series_ft["iFat"], color="#6A4C93", linewidth=1.0, label="iFat")
            ax1.axhline(GAUSSIAN_MAD_OVER_STD, color="black", alpha=0.5, linewidth=0.8, label=f"Gaussiano ({GAUSSIAN_MAD_OVER_STD:.3f})")
            for thr_val, thr_lbl in [
                (thr_1, f"L1={thr_1:.2f}"),
                (thr_2, f"L2={thr_2:.2f}"),
                (thr_3, f"L3={thr_3:.2f}"),
                (thr_4, f"L4={thr_4:.2f}"),
            ]:
                ax1.axhline(thr_val, linestyle="--", linewidth=0.7, alpha=0.5, label=thr_lbl)
            ax1.set_ylabel("iFat")
            ax1.set_title("Índice Fat Tail com níveis de entrada")
            ax1.legend(loc="best", fontsize=8, ncol=3)
            ax1.grid(True, alpha=0.3)

            # Posição target ao longo do tempo
            ax2 = axes_sz[2]
            ax2.fill_between(
                series_ft["data"],
                0,
                series_ft["posicao_target"] * 100,
                color="#D62828",
                alpha=0.3,
                step="post",
            )
            ax2.plot(
                series_ft["data"],
                series_ft["posicao_target"] * 100,
                color="#D62828",
                linewidth=1.2,
                drawstyle="steps-post",
            )
            ax2.set_ylabel("Posição-alvo (%)")
            ax2.set_ylim(-5, 105)
            ax2.set_xlabel("Data")
            ax2.set_title("Posição-alvo conforme o iFat")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_sz, clear_figure=True)

            # ---------- Evolução do patrimônio ----------
            st.markdown("### 💰 Evolução do patrimônio")
            st.caption(
                "Compara a estratégia **escalonada** (comprando só quando iFat sinaliza) "
                "com **buy-and-hold** (comprou tudo no 1º dia e segurou)."
            )

            fig_pat, ax_pat = plt.subplots(figsize=(13, 5))
            ax_pat.plot(
                series_ft["data"],
                series_ft["retorno_acum_%"],
                color="#2E86AB",
                linewidth=1.5,
                label=f"Escalonada ({sz['ret_final_escalonado_%']:+.1f}%)",
            )
            ax_pat.plot(
                series_ft["data"],
                series_ft["retorno_bh_%"],
                color="gray",
                linewidth=1.5,
                linestyle="--",
                label=f"Buy & Hold ({sz['ret_final_buy_hold_%']:+.1f}%)",
            )
            ax_pat.axhline(0, color="black", linewidth=0.7)
            ax_pat.set_ylabel("Retorno acumulado (%)")
            ax_pat.set_xlabel("Data")
            ax_pat.set_title("Escalonada vs Buy & Hold")
            ax_pat.legend(loc="best")
            ax_pat.grid(True, alpha=0.3)
            st.pyplot(fig_pat, clear_figure=True)

            diff = sz["diferenca_%"]
            if diff > 2:
                st.success(
                    f"✅ Estratégia escalonada bateu o buy-and-hold por **{diff:+.2f} pp** no período analisado."
                )
            elif diff < -2:
                st.warning(
                    f"⚠️ Estratégia escalonada ficou **{diff:+.2f} pp** abaixo do buy-and-hold. "
                    "Em mercado em tendência consistente de alta, segurar caixa penaliza o retorno. "
                    "O sistema tende a brilhar em momentos de **crash seguido de recuperação**."
                )
            else:
                st.info(
                    f"ℹ️ Estratégias praticamente empatadas ({diff:+.2f} pp). "
                    "Os dados analisados podem não ter tido crashes suficientes para diferenciar."
                )

            # ---------- Retornos por horizonte ----------
            st.markdown("### 📊 Retorno após cada compra")
            st.caption(
                "Para cada dia em que a posição aumentou (compra), mede o retorno do PU "
                "em N dias. **Retorno ponderado** dá mais peso a compras maiores."
            )

            ret_rows = []
            for h in [30, 60, 90, 180, 365]:
                rp = sz.get(f"ret_medio_ponderado_{h}d_%")
                rs = sz.get(f"ret_medio_simples_{h}d_%")
                bl = sz.get(f"baseline_{h}d_%")
                ex = sz.get(f"excesso_vs_baseline_{h}d_%")
                hr = sz.get(f"hit_rate_{h}d_%")
                n_obs = sz.get(f"n_obs_{h}d")
                if rp is not None:
                    ret_rows.append(
                        {
                            "horizonte": f"{h}d",
                            "ret_ponderado_%": rp,
                            "ret_simples_%": rs,
                            "baseline_%": bl,
                            "excesso_vs_baseline_%": ex,
                            "hit_rate_%": hr,
                            "n_compras": n_obs,
                        }
                    )
            if ret_rows:
                ret_table = pd.DataFrame(ret_rows)
                st.dataframe(ret_table, width="stretch", hide_index=True)
                download_btn(
                    ret_table,
                    f"fat_tail_sizing_bt_{titulo_ft.replace(' ','_')}.csv",
                )

            # ---------- Lista de trades ----------
            with st.expander("📋 Ver todas as compras realizadas"):
                trades = bt_sizing["trades"]
                if not trades.empty:
                    trades_show = trades[
                        ["data", "pu", "iFat", "posicao_target", "posicao_change"]
                    ].copy()
                    trades_show["posicao_target_%"] = (
                        trades_show["posicao_target"] * 100
                    ).round(0)
                    trades_show["compra_%"] = (
                        trades_show["posicao_change"] * 100
                    ).round(0)
                    trades_show = trades_show.drop(
                        columns=["posicao_target", "posicao_change"]
                    )
                    st.dataframe(
                        trades_show, width="stretch", hide_index=True
                    )
                    download_btn(
                        trades_show,
                        f"fat_tail_trades_{titulo_ft.replace(' ','_')}.csv",
                    )


# ===================== TAB Comitê (IA) =====================
with tab_ia:
    st.subheader("🤖 Comitê de Investimento Virtual")
    st.markdown(
        """
        **Um comitê de analistas IA especializados** opina sobre o título que você
        escolher, usando **todos os dados do app** como contexto. Versão atual: MVP
        com 2 agentes (**Analista Técnico** + **Coordenador**).

        ℹ️ *Futuramente pode expandir pra 4 agentes (Macro e Risco).*
        """
    )

    # Verifica dependências
    if not AGENTS_MODULE_AVAILABLE:
        st.error(
            "📦 **Módulo `agents/` não encontrado.** Verifique se a pasta existe "
            "no mesmo diretório do `app.py`."
        )
        st.stop()

    if not ANTHROPIC_AVAILABLE:
        st.error(
            "📦 **Biblioteca `anthropic` não instalada.** Rode:\n\n"
            "```bash\npip install anthropic python-dotenv\n```"
        )
        st.stop()

    import os as _os

    # Tenta pegar a API key do .env
    api_key_env = _os.getenv("ANTHROPIC_API_KEY")

    with st.expander("🔑 Configuração da API key", expanded=(not api_key_env)):
        if api_key_env:
            st.success("✅ API key detectada no `.env`")
            api_key_override = st.text_input(
                "Sobrescrever API key (opcional):",
                type="password",
                help="Só preencha se quiser usar uma chave diferente da do .env",
            )
            effective_api_key = api_key_override if api_key_override else api_key_env
        else:
            st.warning(
                "⚠️ Nenhuma API key no `.env`. Cole abaixo ou crie um `.env` na pasta do app com "
                "`ANTHROPIC_API_KEY=sua-chave`."
            )
            effective_api_key = st.text_input(
                "API key da Anthropic:", type="password"
            )

        if effective_api_key:
            st.caption(
                f"Key: `{effective_api_key[:10]}...{effective_api_key[-4:]}`"
            )

    if not effective_api_key:
        st.info("📝 Informe a API key acima para usar o comitê.")
        st.stop()

    # --- Seleção do título ---
    st.markdown("### 1️⃣ Escolha o título")

    pairs_ia = (
        df[["titulo", "vencimento"]]
        .drop_duplicates()
        .sort_values(["titulo", "vencimento"])
    )
    pairs_ia["label"] = pairs_ia.apply(
        lambda r: f"{r['titulo']} | {r['vencimento'].date()}", axis=1
    )

    sel_ia = st.selectbox(
        "Título para análise do comitê:",
        pairs_ia["label"].tolist(),
        key="ia_sel",
    )

    titulo_ia, venc_ia_str = sel_ia.split(" | ")
    venc_ia = pd.to_datetime(venc_ia_str)

    # --- Coleta contexto automaticamente ---
    st.markdown("### 2️⃣ Contexto coletado do app")

    with st.spinner("Coletando dados do título..."):
        # Linha da tabela de oportunidades
        try:
            opp_cache = compute_opportunities_table(
                df,
                window=90,
                absolute_threshold=0.65,
                main_horizon=90,
                min_events_for_reliability=50,
            )
            opp_row = opp_cache[
                (opp_cache["titulo"] == titulo_ia)
                & (pd.to_datetime(opp_cache["vencimento"]) == venc_ia)
            ]
            opp_row = opp_row.iloc[0] if not opp_row.empty else None
        except Exception:
            opp_row = None

        # Métricas de risco
        s_ia = get_series(df, titulo_ia, venc_ia)
        risk_dict = None
        if "pu" in s_ia.columns and not s_ia["pu"].isna().all():
            try:
                taxa_ia = float(s_ia["taxa"].iloc[-1])
                pu_ia_val = float(s_ia["pu"].iloc[-1])
                anos_ia = max(
                    (venc_ia.normalize() - pd.Timestamp.today().normalize()).days / 365.25,
                    0.01,
                )
                risk_dict = compute_risk_metrics(
                    titulo_ia, taxa_ia, pu_ia_val, anos_ia
                )
            except Exception:
                risk_dict = None

        # Backtest J/Z
        try:
            bt_ia = backtest_mean_reversion(
                df, titulo=titulo_ia, vencimento=venc_ia,
                window_days=365, z_min=1.0,
                horizons=(30, 60, 90, 180),
            )
            bt_summary = bt_ia["summary"]
        except Exception:
            bt_summary = None

        # Fat Tail
        try:
            ft_ia = compute_fat_tail_index(s_ia, window=90, price_col="pu")
            ft_ia_summary = fat_tail_summary(ft_ia) if not ft_ia.empty else None
        except Exception:
            ft_ia_summary = None

        # Monta contexto final
        context = build_context_for_title(
            df,
            titulo_ia,
            venc_ia,
            opportunities_row=opp_row,
            risk_metrics=risk_dict,
            backtest_summary=bt_summary,
            fat_tail_summary=ft_ia_summary,
        )

    with st.expander("📄 Ver contexto coletado (o que o comitê vai analisar)"):
        st.markdown(context)

    # --- Pergunta opcional ---
    st.markdown("### 3️⃣ Sua pergunta (opcional)")
    st.caption(
        "Se deixar em branco, o comitê faz análise livre. "
        "Se quiser foco específico, pergunte: 'devo entrar agora?', 'qual o pior cenário?', "
        "'e se a taxa subir 1%?' etc."
    )

    user_question_ia = st.text_area(
        "Pergunta:",
        value="",
        height=80,
        placeholder="Ex: Devo entrar hoje ou aguardar? Qual o plano de saída?",
        key="ia_question",
    )

    # --- Executar ---
    st.markdown("### 4️⃣ Convocar o comitê")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        executar = st.button("🚀 Convocar comitê", type="primary", key="ia_run")
    with col_btn2:
        st.caption(
            "Custo estimado por consulta: ~R$ 0,05-0,15 (Claude Sonnet). "
            "Usa API key do seu `.env`."
        )

    if executar:
        # Placeholders pra mostrar progresso
        with st.status("🤖 Comitê em sessão...", expanded=True) as status:
            st.write("📊 Analista Técnico avaliando os sinais quantitativos...")
            try:
                result = run_committee(
                    context=context,
                    user_question=user_question_ia,
                    api_key=effective_api_key,
                    agents_to_run=("technical",),
                )
                st.write("🧭 Coordenador sintetizando parecer...")
                status.update(label="✅ Comitê concluído!", state="complete")
            except Exception as e:
                status.update(label=f"❌ Erro: {e}", state="error")
                st.error(f"Falha ao executar comitê: {e}")
                st.stop()

        # --- Mostra resultados ---
        st.markdown("---")
        st.markdown("## 📜 Pareceres do Comitê")

        # Parecer técnico
        if "technical" in result["opinions"]:
            with st.expander("📊 **Parecer do Analista Técnico**", expanded=False):
                st.markdown(result["opinions"]["technical"]["text"])
                st.caption(
                    f"Tokens: {result['opinions']['technical']['input_tokens']} in / "
                    f"{result['opinions']['technical']['output_tokens']} out · "
                    f"Custo: US$ {result['opinions']['technical']['cost_usd']:.4f}"
                )

        # Decisão final do Coordenador — DESTAQUE
        st.markdown("### 🧭 Decisão Final do Coordenador")
        with st.container():
            st.markdown(result["final_decision"]["text"])

        # Custos totais
        st.markdown("---")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Tokens de entrada", f"{result['total_input_tokens']:,}")
        with col_c2:
            st.metric("Tokens de saída", f"{result['total_output_tokens']:,}")
        with col_c3:
            brl_cost = result["total_cost_usd"] * 5.5  # câmbio aproximado
            st.metric(
                "Custo desta consulta",
                f"US$ {result['total_cost_usd']:.4f}",
                delta=f"≈ R$ {brl_cost:.3f}",
            )


# ===================== TAB 7: Guia =====================
with tab7:
    st.subheader("📚 Guia do app — o que é cada coisa e como interpretar")

    st.markdown(
        """
        Este guia te ajuda a **entender cada botão, cada coluna e cada gráfico** do app.
        Clique nos tópicos abaixo para expandir. Pensado para quem está começando em
        renda fixa e também para consulta rápida.
        """
    )

    # =========================================================
    st.markdown("### 🗂️ 1. Fonte dos dados")
    # =========================================================
    with st.expander("📥 De onde vêm os dados?"):
        st.markdown(
            """
            O app baixa o **CSV oficial** do [Tesouro Transparente](https://www.tesourotransparente.gov.br/ckan/dataset/taxas-dos-titulos-ofertados-pelo-tesouro-direto)
            (programa do Tesouro Nacional), atualizado diariamente. Esse é o dado bruto
            que a ANBIMA também usa para calcular os indicadores oficiais.

            Cada linha do CSV é **um título em um dia**. O mesmo título (ex.: Tesouro
            IPCA+ 2035) aparece **uma linha por pregão**, com 4 taxas e 6 PUs diferentes
            (compra/venda × manhã/tarde/base).

            **Atualização:** o app tem cache de 1 hora. Se você quiser forçar uma
            atualização, recarregue a página do navegador (F5).
            """
        )

    # =========================================================
    st.markdown("### 📊 2. Taxa: Compra × Venda · Manhã × Tarde")
    # =========================================================
    with st.expander("💱 Taxa Compra vs Taxa Venda — qual a diferença?"):
        st.markdown(
            """
            No Tesouro Direto, cada título tem **duas taxas** simultâneas:

            | Taxa | Quando se usa | Quem "recebe" |
            |---|---|---|
            | **Taxa Compra** | Quando **você compra** o título do Tesouro | Você (é a taxa que vai render até o vencimento) |
            | **Taxa Venda** | Quando **você vende** o título de volta ao Tesouro antes do vencimento | O Tesouro recompra por esta taxa |

            A **Taxa Venda é um pouco maior** que a Taxa Compra (o PU correspondente é menor) — é o **spread de recompra**, que o Tesouro cobra para te dar liquidez.

            **💡 Qual usar no app?**

            - Se você está **analisando oportunidade de compra** → use **Taxa Compra**. É o que vai definir seu rendimento real. (Default do app.)
            - Se você quer **simular o custo de vender antes do vencimento** → use **Taxa Venda**. É o valor efetivamente realizado se precisar de liquidez.
            - Se você quer ver **quanto o título "vale no mercado"** independente de ponta → use **PU Base Manhã**.
            """
        )

    with st.expander("🌅 Manhã vs 🌇 Tarde — qual escolher?"):
        st.markdown(
            """
            O Tesouro Direto divulga preços em **dois momentos do dia**:

            | Momento | Quando é fixado | Significado |
            |---|---|---|
            | **Manhã** | ~9h30 (abertura do mercado) | Taxa de referência para as operações do dia. É o preço "oficial" usado em backtests acadêmicos. |
            | **Tarde** | ~18h00 (fechamento) | Reflete movimentos do dia (reunião do Copom, leilões, notícias externas). |

            **💡 Qual usar?**

            - **Manhã** é o padrão recomendado. **Séries históricas mais consistentes**, menos ruído de notícia intraday, comparável com a maioria dos estudos.
            - **Tarde** só se você quer capturar especificamente movimentos intraday (ex.: dia de Copom).

            O app usa **Manhã por padrão** e você pode trocar na sidebar.

            **⚠️ Cuidado:** não misture manhã e tarde em análises longas — os ruídos são diferentes e podem criar falsos sinais de reversão.
            """
        )

    # =========================================================
    st.markdown("### 💰 3. PU: Compra · Venda · Base")
    # =========================================================
    with st.expander("🧾 O que é PU e qual versão escolher?"):
        st.markdown(
            r"""
            **PU = Preço Unitário**, ou seja, **quanto custa uma unidade do título em R\$** naquele momento.

            A relação matemática com a taxa é direta: **PU alto ⇔ taxa baixa** (e vice-versa).
            Para IPCA+/prefixado sem cupom:

            $$
            PU = \frac{VNA}{(1 + \text{taxa})^{\text{anos até vencimento}}}
            $$

            onde VNA é o Valor Nominal Atualizado (R\$ 1.000 no caso do Prefixado, ou
            1.000 × fator IPCA para IPCA+).

            **As 3 versões disponíveis no CSV:**

            | PU | Quando usar | Observação |
            |---|---|---|
            | **PU Compra** | Se o app vai simular **você comprando** (caso mais comum) | Default do app |
            | **PU Venda** | Para simular **resgate antecipado / venda antes do vencimento** | Um pouco **menor** que PU Compra (você vende mais barato do que compra) |
            | **PU Base** | Preço de **referência neutro**, sem spread bid/ask | Usado em backtests acadêmicos e em estudos da ANBIMA |

            **💡 Regra prática:**

            - 🛒 **Análise de compra / carrego** → PU Compra
            - 💸 **Análise de liquidez / saída antecipada** → PU Venda
            - 📊 **Séries históricas / backtests / Fat Tail** → PU Base (o mais "limpo")

            O app usa **PU Compra Manhã** por padrão, mas para análises estatísticas
            (como o iFat na aba Fat Tails) considere trocar para **PU Base Manhã** na sidebar.
            """
        )

    # =========================================================
    st.markdown("### 🏆 4. Como ler cada aba do app")
    # =========================================================

    with st.expander("🎯 Aba Oportunidades — a tela de decisão"):
        st.markdown(
            """
            **O que faz:** combina em uma única tela o **histórico do sinal** + o
            **status atual** de cada título, com um **veredito automático** classificando
            onde vale usar o Fat Tail.

            **Colunas da tabela:**

            | Coluna | Significado |
            |---|---|
            | `iFat_atual` | Valor atual do índice MAD/STD. Gaussiano ≈ 0,798. Abaixo = stress. |
            | `posicao_alvo_%` | Quanto do capital alocar agora, via position sizing escalonado. |
            | `excesso_Nd_%` | Ganho histórico médio vs baseline no horizonte escolhido (ex. 90d). |
            | `hit_rate_%` | % das vezes em que o sinal deu resultado positivo. |
            | `n_eventos_hist` | Quantas vezes o sinal disparou historicamente neste título. |
            | `veredito` | Classificação automática (ver abaixo). |
            | `recomendacao` | Ação sugerida combinando veredito + status atual. |

            **Critérios do veredito automático:**

            | Veredito | Condição |
            |---|---|
            | ✅ **FORTE** | Excesso > 2%, hit rate > 60%, ≥ 50 eventos históricos |
            | 🟢 **OK** | Excesso > 0.5%, hit rate > 55% |
            | 🟡 **FRACO** | Sinal positivo mas pequeno — use com cautela |
            | ❌ **EVITAR** | Excesso negativo OU hit rate < 45% — sinal contrário |
            | 📊 **SEM DADOS** | Menos de 10 eventos — estatística insuficiente |

            **Como usar na prática:**

            1. **Filtre** por vereditos FORTE e OK
            2. **Ordene** por posição-alvo maior (os que já estão sinalizando entrada)
            3. **Seção "Ação recomendada AGORA"**: lista os 5 títulos mais prontos, com
               valor de posição sugerido e contexto histórico

            **💡 Por que essa aba é especial:**
            É a **única tela** que combina três perguntas críticas de uma vez:
            - O sinal funciona neste título? (histórico)
            - Está disparando agora? (iFat atual)
            - Quanto devo alocar? (position sizing)

            Sem essa síntese, você precisaria abrir 3-4 abas para chegar à mesma conclusão.
            """
        )

    with st.expander("🏆 Aba 1 — Scanner"):
        st.markdown(
            """
            **O que faz:** rankeia todos os títulos disponíveis pelo potencial de
            oportunidade (taxa historicamente alta = possível ponto de entrada).

            **Como funciona o score:**

            Para cada título, calcula em janelas de 180/365/730 dias:
            - **J (quartil):** em qual quartil da distribuição histórica da própria taxa do título ela está hoje.
              - `J1` = taxa baixa (histórico) → título provavelmente "caro"
              - `J4` = taxa alta (histórico) → título provavelmente "barato"
            - **Z (z-score):** quantos desvios-padrão a taxa atual está da média histórica.
              - `Z = +1.5` significa 1,5 desvios acima da média → pressão de compra.

            **Score composto:**
            ```
            score = 2·pts(J730) + 1·pts(J365) + 1.5·Z730 + 0.8·Z365 + 0.5·(concordância 12m/24m)
            ```

            **Como usar na prática:**
            1. Filtre por família (IPCA+, Prefixado, etc.).
            2. Ordene por score descendente.
            3. Títulos com **J4 em ambas as janelas E Z ≥ 1.0** são candidatos fortes.
            4. **Ative o filtro Fat Tail** para adicionar a dimensão de "pânico" — combinação J4+Z+iFat é mais confiável que cada sinal sozinho.

            **⚠️ Cuidado:** score alto **não é recomendação de compra**. É só um rankeador estatístico. Sempre cheque a aba Backtest para verificar se esse sinal historicamente funcionou **para esse título específico**.
            """
        )

    with st.expander("🧪 Aba 2 — Backtest (validação)"):
        st.markdown(
            """
            **O que faz:** pega um título específico e testa: "se eu tivesse comprado
            toda vez que J4+Z disparou no passado, quanto teria rendido?"

            **Cards de resumo:**
            - **Retorno médio PU (90d)**: média dos retornos em 90 dias após os sinais.
            - **Mediana (90d)**: valor do meio (menos afetado por outliers).
            - **Hit rate PU↑ (90d)**: % dos sinais em que o PU subiu depois.
            - **Sharpe-like**: retorno médio ÷ desvio dos retornos. Acima de 0.5 é bom.

            **Como interpretar os gráficos:**

            - **Série histórica de Taxa:** se você vê **picos** (taxa sobe abruptamente) seguidos de **quedas** (reversão), a hipótese de reversão à média tem base empírica para esse título.
            - **Série histórica de PU:** espelho invertido da taxa. Pânicos = vales no PU.
            - **Histograma de retornos 90d:** se a massa está à direita do 0 (linha vermelha) e a média (linha verde) é positiva, o sinal **funcionou**.

            **Regras de bolso:**
            - Hit rate > 55% + retorno médio positivo = sinal útil ✅
            - Hit rate < 45% = sinal enganoso; evite usar ❌
            - Hit rate 45-55% = zona cinza; peça confirmação de outra aba
            """
        )

    with st.expander("📐 Aba 3 — Risco (Duration)"):
        st.markdown(
            r"""
            **O que faz:** quantifica **o risco de taxa** do título — quanto o PU
            balança para cada 1% de variação na taxa.

            **Os 4 números principais:**

            | Métrica | O que significa | Intuição |
            |---|---|---|
            | **Duration (Macaulay)** | Tempo médio ponderado para receber o fluxo do título (em anos) | Zero-cupom: = prazo até o vencimento. Com cupom: menor que o prazo. |
            | **Modified Duration** | % que o PU muda se a taxa mudar 1% | MD = 5 → taxa +1% faz o PU cair ~5% |
            | **Convexidade** | Correção não-linear da Duration | Quanto maior, mais o PU sobe em queda de taxa e menos cai em alta |
            | **DV01** | Quanto o PU muda (em R\$) a cada +0,01% (1 bp) na taxa | Posição de R\$ 100 mil com DV01 = R\$ 50 → perde R\$ 50 por 1 bp de alta |

            **Aproximação de 2ª ordem (a tabela de sensibilidade usa isso):**
            $$
            \frac{\Delta PU}{PU} \approx -\text{MD} \cdot \Delta y + \tfrac{1}{2} \cdot \text{Convex} \cdot (\Delta y)^2
            $$

            **Como interpretar o gráfico de sensibilidade:**
            - Eixo X: variação da taxa em bps (1 bp = 0,01%)
            - Eixo Y: variação esperada do PU em %
            - **Se a curva parece uma linha reta**: duration domina (prazo curto).
            - **Se a curva é encurvada**: convexidade importa (prazo longo).

            **Modo comparativo (scatter):**
            - Eixo X: prazo até vencimento
            - Eixo Y: Modified Duration
            - **Tamanho da bolha**: DV01 (quanto maior, mais risco em R\$)
            - Use para comparar rapidamente risco relativo entre vencimentos da mesma família.

            **💡 Regras rápidas:**
            - IPCA+ 2029 (curto): MD ~3 → risco baixo
            - IPCA+ 2045 (longo): MD ~15 → risco **5× maior**. Choques de 1% = ±15% no PU.
            - Se você não tem estômago para oscilação de 15-20%, evite o trecho longo.
            """
        )

    with st.expander("🌐 Aba 4 — Curvas & Inflação"):
        st.markdown(
            r"""
            **O que faz:** mostra a **estrutura a termo** (curva de juros) e a
            **inflação implícita** que o mercado está precificando.

            **Gráfico da curva de juros:**
            - Eixo X: prazo (anos até vencimento)
            - Eixo Y: taxa atual do título
            - **Curva positivamente inclinada**: taxa de prazos longos > curtos (normal, com prêmio de prazo).
            - **Curva invertida**: taxa de prazos curtos > longos → mercado espera Selic caindo no futuro. Historicamente associado a ciclos de afrouxamento monetário.
            - **Curva "flat"**: prazos todos parecidos → mercado indeciso sobre direção dos juros.

            **Inflação implícita (Pré vs IPCA+):**

            Para cada par de vencimentos próximos, calcula:
            $$
            \text{IPCA implícito} = \frac{1 + \text{taxa pré}}{1 + \text{taxa real IPCA+}} - 1
            $$

            **Como ler:**
            - Se mercado precifica IPCA em **5% a.a.** e você acredita que vai ser **4%** → **compre Prefixado** (render mais que IPCA+).
            - Se mercado precifica **5%** e você acredita em **6%** → **compre IPCA+** (a correção pela inflação compensa).
            - Se as duas alternativas parecem equivalentes dada sua expectativa → use como decisão em outros critérios (duration, prazo, liquidez).

            **💡 Esta é a pergunta central de renda fixa no Brasil.** Se você está
            alocando entre Pré e IPCA+, é essa tabela que importa mais que qualquer
            outra análise do app.
            """
        )

    with st.expander(r"📈 Aba 5 — Cenários (R\$)"):
        st.markdown(
            r"""
            **3 sub-abas**, cada uma responde uma pergunta diferente:

            **1. PU futuro (MtM):** "e se a taxa for X% em um ano, quanto meu PU vale?"
            - Usa regressão linear histórica PU ~ taxa na janela escolhida.
            - Cenários stress/base/otimista.
            - O gráfico de dispersão mostra a relação histórica; a linha vermelha é a regressão.
            - ⚠️ **Regressão linear é uma aproximação.** Para choques grandes (± 2%), a relação real é **convexa** (veja aba 3 sobre Convexidade). A regressão **subestima ganhos em quedas de taxa** e superestima perdas em altas.

            **2. Carrego até o vencimento (IPCA+):**
            - Simula: "compro R\$ X, carrego até vencer, quanto tenho?"
            - Assume: PU evolui por `(1+real)^t × (1+IPCA)^t`.
            - Aplica **custódia** (B3, 0,20% a.a.) e **IR regressivo** (15-22,5%).
            - **DCA (aportes periódicos):** se você vai aportar R\$ 500/mês, insira esse valor. O gráfico mostra separadamente o que **você aportou** (linha pontilhada) e o **valor total com rendimentos** (linhas coloridas).
            - 3 cenários de IPCA (baixo/base/alto) para ver range de resultados.

            **3. Carrego vs Venda antecipada:**
            - Responde: "vale mais a pena segurar até o fim ou vender em N anos?"
            - Simula os dois cenários lado a lado (mesmo horizonte de tempo, mesmo IR).
            - **Leitura:** se venda > carrego, o mercado está te oferecendo um prêmio para você abrir mão do carrego. Se carrego > venda, o carrego está mais gordo que a marcação a mercado sugere.
            """
        )

    with st.expander("🐘 Aba 6 — Fat Tails (Taleb)"):
        st.markdown(
            r"""
            **O que faz:** detecta **regimes de cauda gorda** nos retornos do PU.
            Quando o mercado está em pânico, os retornos saem do padrão gaussiano —
            aparecem dias muito fortes (negativos e positivos). O índice MAD/STD captura isso.

            **Matemática:**
            - **MAD** (Mean Absolute Deviation): desvio absoluto médio dos retornos
            - **STD**: desvio-padrão
            - **iFat = MAD/STD**
            - Para distribuição **Normal (gaussiana)**: iFat ≈ **0,7979** (= √(2/π))
            - iFat **abaixo** do gaussiano → caudas mais gordas que o normal → regime de stress

            **Como interpretar os gráficos:**

            **Gráfico PU + iFat:**
            - Gráfico de cima (PU em azul): os **pontos vermelhos** marcam os dias em que o iFat cruzou o limiar de cauda gorda.
            - Gráfico de baixo (iFat em roxo):
              - **Linha laranja tracejada** = média móvel do próprio iFat
              - **Linha vermelha pontilhada** = banda inferior (`média - desvio`)
              - **Linha preta horizontal** = referência gaussiana (0,7979)
              - **Linha cinza** = limiar absoluto (ex.: 0,65)
            - **Zonas de entrada** = quando o iFat fica abaixo **das duas linhas tracejadas/pontilhadas** simultaneamente.

            **Histograma do iFat:**
            - A distribuição costuma ser **bimodal**: uma moda perto de 0,80 (mercado normal) e outra abaixo (regime de stress).
            - **Linha preta** = gaussiano, **linha vermelha** = iFat atual, **linha cinza** = limiar.
            - Se o iFat atual estiver na moda inferior → você está em regime de pânico.

            **Backtest da estratégia:**
            - **Retorno médio** após o sinal vs **baseline** (média sem filtro).
            - **Excesso positivo** = sinal adiciona valor sobre comprar qualquer dia.
            - ✅ Excesso > 0 em múltiplos horizontes = sinal robusto.
            - ⚠️ Excesso negativo = estratégia NÃO funciona para esse título.

            **💡 Calibração:**
            Rode o **notebook de validação** (`validacao_fat_tail.ipynb` no repo) para
            descobrir qual `(janela, limiar)` funciona melhor nos dados reais antes de
            usar como default.

            ---

            ### 📊 Position Sizing Escalonado (seção dentro da aba 6)

            **O problema do sinal binário:** a regra `iFat < banda móvel E iFat < 0.65`
            tende a **atrasar a entrada**. Observando o gráfico historicamente, você
            vê o iFat despencando DURANTE o crash — mas os pontos vermelhos só aparecem
            no **fim** do crash, quando a banda móvel "pegou" o movimento. Nesse
            intervalo, o PU já reagiu e a oportunidade passou.

            **Solução:** posição proporcional à profundidade do stress. Em vez de
            "comprar tudo ou nada", construa a posição em fatias:

            | Faixa de iFat | Posição recomendada | Interpretação |
            |---|---|---|
            | ≥ 0.75 | 0% | mercado normal |
            | 0.70 - 0.75 | 25% | leve pressão |
            | 0.65 - 0.70 | 50% | stress moderado |
            | 0.60 - 0.65 | 75% | stress forte |
            | < 0.60 | 100% | pânico |

            **Vantagens:**
            - **Melhor preço médio**: você compra na descida, não só no fundo.
            - **Proteção contra falsos sinais**: se iFat toca 0.72 e volta, você fica
              com só 25% investido — o "falso" custa pouco.
            - **Plano executável**: não precisa acertar o fundo perfeito.

            **Como ler os gráficos da seção:**

            - **PU com pontos verdes/vermelhos**: verde = dia de compra, vermelho = dia
              de redução. **Tamanho do ponto = quanto foi comprado/reduzido**.
            - **Escadinha abaixo**: mostra a posição-alvo ao longo do tempo (0 a 100%).
              Ela sobe em degraus quando o iFat quebra um novo nível.
            - **Curva de patrimônio**: compara a estratégia escalonada com buy-and-hold.
              Em mercado lateral ou com crashes → escalonada ganha. Em tendência
              sustentada de alta → buy-and-hold ganha (segurar caixa penaliza).

            **⚠️ Esperado:** em **alguns** títulos e **alguns** períodos, a estratégia
            escalonada **PERDE** para buy-and-hold. Isso é natural — não existe
            estratégia que ganhe sempre. A pergunta certa é: "**em que tipo de regime**
            ela ganha?" (resposta: crashes seguidos de recuperação forte — 2020, 2022).
            """
        )

    # =========================================================
    st.markdown("### 🔤 5. Glossário rápido")
    # =========================================================
    with st.expander("📖 Termos usados no app (do A ao Z)"):
        st.markdown(
            r"""
            | Termo | Definição |
            |---|---|
            | **bp (basis point)** | 1 centésimo de 1%. 100 bps = 1%. |
            | **Convexidade** | Mede a curvatura da relação PU×taxa. Títulos longos têm mais. |
            | **Cupom** | Pagamento periódico de juros (semestral em NTN-F e NTN-B c/ Juros Semestrais). |
            | **DV01** | Variação do PU em R\$ para +1 bp na taxa. |
            | **Duration** | Prazo médio ponderado dos fluxos (em anos). |
            | **ETTJ** | Estrutura a Termo da Taxa de Juros (curva de juros). |
            | **Fat Tail** | Cauda gorda; retornos extremos mais frequentes que no gaussiano. |
            | **iFat** | Índice MAD/STD usado para detectar fat tails. |
            | **IPCA+** | Tesouro atrelado ao IPCA (NTN-B). Paga inflação + juro real. |
            | **J1-J4** | Quartis da taxa histórica do título. J4 = taxa atual é das maiores. |
            | **LFT** | Tesouro Selic (pós-fixado pela Selic). |
            | **LTN** | Tesouro Prefixado (zero-cupom). |
            | **MAD** | Mean Absolute Deviation — desvio absoluto médio. |
            | **Modified Duration** | Duration / (1+taxa). % que o PU muda por 1% de taxa. |
            | **NTN-B** | Tesouro IPCA+ (com ou sem Juros Semestrais). |
            | **NTN-F** | Tesouro Prefixado com Juros Semestrais. |
            | **PU** | Preço Unitário do título em R\$. |
            | **Selic** | Taxa básica de juros fixada pelo Copom. Referência do LFT. |
            | **Sharpe** | Retorno médio / desvio-padrão. Mede qualidade do retorno ajustado ao risco. |
            | **STD** | Standard Deviation — desvio-padrão. |
            | **VNA** | Valor Nominal Atualizado (R\$ 1.000 base, corrigido por IPCA se NTN-B). |
            | **Z-score** | (valor − média) / desvio. Mede quantos desvios da média está um ponto. |
            """
        )

    # =========================================================
    st.markdown("### 🗺️ 6. Fluxo de uso recomendado")
    # =========================================================
    with st.expander("🚀 Do zero até uma decisão — passo a passo"):
        st.markdown(
            """
            Se você está abrindo o app pela primeira vez para decidir onde alocar,
            a sequência mais eficiente é:

            **1. 🎯 Abra a aba "Oportunidades" PRIMEIRO**
            - É a **tela-síntese**. Mostra todos os títulos com veredito automático.
            - Procure os que têm veredito ✅ FORTE ou 🟢 OK **E** posição-alvo > 0%.
            - Se nada estiver ativo → aguarde. Não force entrada.

            **2. 🌐 Aba "Curvas & Inflação"**
            - Veja o formato da curva de juros (inclinada? invertida?).
            - Olhe a inflação implícita. Compare com sua expectativa pessoal.
            - Decisão: **Pré ou IPCA+?**

            **3. 🏆 Vá para o Scanner**
            - Filtre pela família que você decidiu (Pré ou IPCA+).
            - Anote os 2-3 títulos com maior score que coincidem com os FORTES em Oportunidades.

            **4. 📐 Aba Risco**
            - Compare os candidatos no modo **multi-título**.
            - Escolha o que combina com seu horizonte e apetite de risco.

            **5. 🧪 Aba Backtest**
            - Teste se o sinal J4+Z historicamente funcionou **para esse título específico**.
            - Se hit rate > 55% e retorno médio positivo → sinal validado.

            **6. 🐘 Aba Fat Tails**
            - Confirme visualmente o status atual do iFat.
            - Use o **Position Sizing escalonado** para decidir o tamanho de entrada.

            **7. 📈 Aba Cenários**
            - Simule carrego com DCA se aplicável.
            - Faça stress de taxa para ver o pior caso em MtM.

            **8. 🎯 Decisão final**
            - Veredito FORTE + posição-alvo > 50% + J4/Z concorda = entrada confiante.
            - Veredito OK + 1-2 concordâncias = entre com posição menor.
            - Sinais divergentes = espere ou vá para Selic.

            **⚠️ Sempre lembre:**
            - Este é um **estudo quantitativo**. O app NÃO recomenda investimentos.
            - Diversifique por prazo (curto+médio+longo) para não ficar sensível a um único regime de taxa.
            - IR regressivo: resgate antes de 181 dias paga 22,5% sobre o ganho. Não faça market timing de curto prazo em RF sem conta corrente do impacto.
            """
        )
