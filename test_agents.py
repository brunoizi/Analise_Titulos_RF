"""
Script de teste dos agentes.

Rode esse arquivo pra testar se a integração com a API da Anthropic funciona
ANTES de abrir o Streamlit. Valida:
- Se o .env está sendo lido
- Se a biblioteca anthropic responde
- Se os prompts funcionam
- Quanto custa uma consulta típica

Uso:
    python test_agents.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Carrega .env se existir
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    print("❌ ANTHROPIC_API_KEY não encontrada no .env ou env var")
    print("   Crie um arquivo .env com:")
    print("   ANTHROPIC_API_KEY=sk-ant-api03-xxxxx")
    sys.exit(1)

print(f"✅ API key detectada: {API_KEY[:10]}...{API_KEY[-4:]}")
print()

# Importa agentes
try:
    from agents.orchestrator import build_context_for_title, run_committee
except ImportError as e:
    print(f"❌ Erro ao importar agents: {e}")
    sys.exit(1)

print("✅ Módulo agents importado com sucesso")
print()

# Monta um contexto fake realista (simulando dados do app)
print("📦 Montando contexto de teste...")

fake_context = """# TÍTULO ANALISADO: Tesouro IPCA+ com Juros Semestrais 2045
**Vencimento:** 2045-05-15
**Anos até vencimento:** 19.07
**Data de referência:** 2026-04-17
**Taxa atual (% a.a.):** 7.6274
**PU atual (R$):** 3,483.68

## HISTÓRICO DE TAXA
- **Últimos 30 dias:** min 7.50 / média 7.55 / máx 7.63 / atual 7.63
- **Últimos 180 dias:** min 7.25 / média 7.47 / máx 7.63 / atual 7.63
- **Últimos 365 dias:** min 6.98 / média 7.36 / máx 7.63 / atual 7.63
- **Últimos 730 dias:** min 6.75 / média 7.14 / máx 7.63 / atual 7.63

## SINAIS DO SCANNER (Oportunidades)
- **Veredito do app:** ✅ FORTE
- **iFat atual:** 0.617
- **Posição-alvo sugerida pelo sizing escalonado:** 75%
- **excesso_90d_%:** +2.60%
- **Hit rate histórico:** 72.0%
- **N° de eventos históricos:** 524

## MÉTRICAS DE RISCO (Duration)
- **Duration Macaulay:** 12.50 anos
- **Modified Duration:** 11.80
- **Convexidade:** 182.00
- **DV01 (R$ por 1 bp):** R$ 4.13
- **Interpretação:** choque de +1% na taxa ≈ -11.8% no PU.

## BACKTEST J/Z (reversão à média)
- **N° de eventos:** 125
- **Ret médio PU em 90d após sinal:** +2.85%  (hit rate: 72.4%)
- **Ret médio PU em 180d após sinal:** +4.10%  (hit rate: 68.0%)

## ÍNDICE FAT TAIL (iFat)
- **Convenção:** mad_over_std
- **Valor atual:** 0.6170
- **Gaussiano de referência:** 0.7979
- **Média histórica:** 0.7450
- **% tempo em cauda gorda:** 18.0%
"""

print("✅ Contexto montado (~350 tokens)")
print()

# Pergunta do usuário
user_question = "Devo entrar agora neste título? Qual o plano de saída?"
print(f"❓ Pergunta de teste: '{user_question}'")
print()

print("🚀 Convocando comitê (pode demorar 10-20 segundos)...")
print("-" * 60)

try:
    result = run_committee(
        context=fake_context,
        user_question=user_question,
        api_key=API_KEY,
        agents_to_run=("technical",),
    )
except Exception as e:
    print(f"❌ Erro ao rodar comitê: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("📊 PARECER DO ANALISTA TÉCNICO")
print("=" * 60)
print(result["opinions"]["technical"]["text"])

print()
print("=" * 60)
print("🧭 DECISÃO FINAL DO COORDENADOR")
print("=" * 60)
print(result["final_decision"]["text"])

print()
print("=" * 60)
print("💰 CUSTOS")
print("=" * 60)
print(f"Tokens de entrada:  {result['total_input_tokens']:,}")
print(f"Tokens de saída:    {result['total_output_tokens']:,}")
print(f"Custo total:        US$ {result['total_cost_usd']:.4f}")
print(f"                  ≈ R$ {result['total_cost_usd'] * 5.5:.3f}")
print()
print("✅ Teste concluído com sucesso!")
print()
print("👉 Próximo passo: rode 'streamlit run app.py' e abra a aba 🤖 Comitê")
