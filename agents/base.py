"""
Classe base para os agentes do Comitê de Investimento Virtual.

Cada agente é uma chamada à API da Anthropic com:
- Um system prompt especializado (a "personalidade" do agente)
- Um contexto estruturado (os dados do app)
- Uma pergunta do usuário ou uma tarefa específica
"""

from __future__ import annotations
import os
from typing import Any

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class Agent:
    """
    Agente base. Cada agente especializado (Técnico, Macro, Risco, Coordenador)
    herda desta classe e define seu próprio system prompt.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2000,
        api_key: str | None = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens

        # API key vem do parâmetro, ou do env, ou de st.secrets (vazio = erro)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Biblioteca 'anthropic' não instalada. "
                "Rode: pip install anthropic"
            )

        if not self.api_key:
            raise ValueError(
                "API key da Anthropic não encontrada. "
                "Defina ANTHROPIC_API_KEY no .env ou passe como parâmetro."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze(self, context: str, question: str = "") -> dict[str, Any]:
        """
        Executa a análise.

        Parâmetros:
        - context: texto com todos os dados relevantes (tabelas, números, histórico).
        - question: pergunta específica do usuário (opcional).

        Retorna dict com:
        - text: resposta textual do agente
        - input_tokens / output_tokens: consumo
        - cost_usd: custo estimado em USD
        """
        # Monta a mensagem
        if question:
            user_message = (
                f"## CONTEXTO DOS DADOS\n\n{context}\n\n"
                f"## PERGUNTA DO USUÁRIO\n\n{question}\n\n"
                f"Responda estritamente como um **{self.role}**, "
                f"conforme as regras do seu papel."
            )
        else:
            user_message = (
                f"## CONTEXTO DOS DADOS\n\n{context}\n\n"
                f"Emita seu parecer como **{self.role}**, "
                f"conforme as regras do seu papel."
            )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extrai texto
        text = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )

        # Custo estimado (Claude Sonnet 4.5: $3/MTok input, $15/MTok output)
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost_usd = (in_tok * 3.0 / 1_000_000) + (out_tok * 15.0 / 1_000_000)

        return {
            "text": text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost_usd,
            "agent": self.name,
        }
