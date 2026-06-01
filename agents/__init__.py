"""
Módulo de agentes de IA para o Comitê de Investimento Virtual.

Uso básico:
    from agents.orchestrator import build_context_for_title, run_committee

    ctx = build_context_for_title(df, titulo, vencimento, opportunities_row=row)
    result = run_committee(ctx, user_question="Devo entrar agora?")
    print(result["final_decision"]["text"])
"""

from .base import Agent, ANTHROPIC_AVAILABLE, DEFAULT_MODEL
from .technical import TechnicalAnalyst
from .coordinator import Coordinator
from .orchestrator import build_context_for_title, run_committee

__all__ = [
    "Agent",
    "TechnicalAnalyst",
    "Coordinator",
    "build_context_for_title",
    "run_committee",
    "ANTHROPIC_AVAILABLE",
    "DEFAULT_MODEL",
]
