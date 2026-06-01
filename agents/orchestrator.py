"""
Orquestrador do Comitê: coleta contexto do app e coordena os agentes.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from .technical import TechnicalAnalyst
from .coordinator import Coordinator


def build_context_for_title(
    df: pd.DataFrame,
    titulo: str,
    vencimento: pd.Timestamp,
    opportunities_row: pd.Series | None = None,
    risk_metrics: dict | None = None,
    backtest_summary: dict | None = None,
    fat_tail_summary: dict | None = None,
) -> str:
    """
    Monta um texto estruturado com TODOS os dados relevantes sobre um título
    específico, pronto para ser injetado no prompt dos agentes.

    Esse é o 'único input fonte' — tudo o que os agentes sabem vem daqui.
    Garante que eles não inventam dados.
    """
    parts = [f"# TÍTULO ANALISADO: {titulo}"]
    parts.append(f"**Vencimento:** {vencimento.date() if hasattr(vencimento, 'date') else vencimento}")

    hoje = pd.Timestamp.today().normalize()
    try:
        anos_ate = (vencimento - hoje).days / 365.25
        parts.append(f"**Anos até vencimento:** {anos_ate:.2f}")
    except Exception:
        pass

    # Taxa e PU atuais
    d = df[(df["titulo"] == titulo) & (df["vencimento"] == vencimento)].sort_values("data")
    if not d.empty:
        ult = d.iloc[-1]
        parts.append(f"**Data de referência:** {ult['data'].date()}")
        parts.append(f"**Taxa atual (% a.a.):** {ult['taxa']:.4f}")
        if "pu" in d.columns and pd.notna(ult.get("pu")):
            parts.append(f"**PU atual (R$):** {ult['pu']:,.2f}")

        # Histórico resumido: min/max/média de taxa em diferentes janelas
        parts.append("\n## HISTÓRICO DE TAXA")
        for label, days in [("30 dias", 30), ("180 dias", 180), ("365 dias", 365), ("730 dias", 730)]:
            start = ult["data"] - pd.Timedelta(days=days)
            w = d[(d["data"] >= start) & (d["data"] <= ult["data"])]["taxa"].dropna()
            if len(w) > 0:
                parts.append(
                    f"- **Últimos {label}:** min {w.min():.3f} / "
                    f"média {w.mean():.3f} / máx {w.max():.3f} / "
                    f"atual {ult['taxa']:.3f}"
                )

    # Sinais da aba Oportunidades
    if opportunities_row is not None:
        parts.append("\n## SINAIS DO SCANNER (Oportunidades)")
        parts.append(f"- **Veredito do app:** {opportunities_row.get('veredito', 'n/d')}")
        parts.append(f"- **iFat atual:** {opportunities_row.get('iFat_atual', 'n/d')}")
        parts.append(f"- **Posição-alvo sugerida pelo sizing escalonado:** {opportunities_row.get('posicao_alvo_%', 0):.0f}%")

        # Excesso e hit rate (nomes das colunas variam com o horizonte)
        for col in opportunities_row.index:
            if col.startswith("excesso_") and col.endswith("_%"):
                val = opportunities_row[col]
                if pd.notna(val):
                    parts.append(f"- **{col}:** {val:+.2f}%")
        if "hit_rate_%" in opportunities_row.index:
            parts.append(f"- **Hit rate histórico:** {opportunities_row['hit_rate_%']:.1f}%")
        if "n_eventos_hist" in opportunities_row.index:
            parts.append(f"- **N° de eventos históricos:** {opportunities_row['n_eventos_hist']}")

    # Métricas de risco
    if risk_metrics:
        parts.append("\n## MÉTRICAS DE RISCO (Duration)")
        parts.append(f"- **Duration Macaulay:** {risk_metrics.get('duration_macaulay_a', 0):.2f} anos")
        parts.append(f"- **Modified Duration:** {risk_metrics.get('duration_modified_a', 0):.2f}")
        parts.append(f"- **Convexidade:** {risk_metrics.get('convexidade_a', 0):.2f}")
        parts.append(f"- **DV01 (R$ por 1 bp):** R$ {risk_metrics.get('dv01_R$', 0):.2f}")
        parts.append(f"- **Interpretação:** choque de +1% na taxa ≈ {-risk_metrics.get('duration_modified_a', 0):.1f}% no PU.")

    # Backtest J/Z
    if backtest_summary:
        parts.append("\n## BACKTEST J/Z (reversão à média)")
        parts.append(f"- **N° de eventos:** {backtest_summary.get('n_events', 0)}")
        for h in [30, 60, 90, 180]:
            key = f"mean_ret_pu_{h}d"
            if key in backtest_summary:
                parts.append(
                    f"- **Ret médio PU em {h}d após sinal:** "
                    f"{backtest_summary[key] * 100:+.2f}%  "
                    f"(hit rate: {backtest_summary.get(f'hit_pu_up_{h}d', 0) * 100:.1f}%)"
                )

    # Fat Tail detalhado
    if fat_tail_summary:
        parts.append("\n## ÍNDICE FAT TAIL (iFat)")
        parts.append(f"- **Convenção:** {fat_tail_summary.get('convention', 'mad_over_std')}")
        parts.append(f"- **Valor atual:** {fat_tail_summary.get('valor_atual', 0):.4f}")
        parts.append(f"- **Gaussiano de referência:** {fat_tail_summary.get('gaussian_ref', 0):.4f}")
        parts.append(f"- **Média histórica:** {fat_tail_summary.get('media', 0):.4f}")
        parts.append(f"- **% tempo em cauda gorda:** {fat_tail_summary.get('pct_em_cauda_gorda', 0) * 100:.1f}%")

    return "\n".join(parts)


def run_committee(
    context: str,
    user_question: str = "",
    api_key: str | None = None,
    agents_to_run: list[str] = ("technical",),
) -> dict:
    """
    Roda o comitê: agentes emitem pareceres e o Coordenador sintetiza.

    Retorna dict com:
    - opinions: dict {nome_agente: parecer}
    - final_decision: decisão do coordenador
    - total_cost_usd
    - total_tokens_in / total_tokens_out
    """
    opinions = {}
    total_in, total_out, total_cost = 0, 0, 0.0

    # 1. Analista Técnico
    if "technical" in agents_to_run:
        tech = TechnicalAnalyst(api_key=api_key)
        result = tech.analyze(context, user_question)
        opinions["technical"] = result
        total_in += result["input_tokens"]
        total_out += result["output_tokens"]
        total_cost += result["cost_usd"]

    # 2. (Futuro: Macro e Risco entrariam aqui)

    # 3. Coordenador sintetiza
    coord = Coordinator(api_key=api_key)
    final = coord.consolidate(
        technical_opinion=opinions.get("technical", {}).get("text", ""),
        context_summary=context,
        user_question=user_question,
    )
    total_in += final["input_tokens"]
    total_out += final["output_tokens"]
    total_cost += final["cost_usd"]

    return {
        "opinions": opinions,
        "final_decision": final,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_cost,
    }
