# 🔬 Validação Empírica do Índice Fat Tail

Notebook de análise rigorosa: **o sinal Fat Tail é útil para comprar Tesouro Direto?**

## 📦 Arquivos

- **`validacao_fat_tail.ipynb`** (na raiz do repo) — notebook limpo, pronto para executar.
- **`engine.py`** (na raiz do repo) — engine quantitativa usada pelo notebook e pelo app.

## 🚀 Como rodar

```bash
# no seu ambiente Python com jupyter
pip install pandas numpy matplotlib requests jupyter

# na raiz do repositório:
jupyter notebook validacao_fat_tail.ipynb
```

A primeira célula tenta baixar dados reais do Tesouro Transparente. Se falhar (rede restrita), ela tenta usar um CSV local como fallback.

## 🎯 O que o notebook responde

O notebook roda **3 níveis de análise**:

1. **Um título específico** — você escolhe um IPCA+ longo e vê todos os eventos Fat Tail historicamente, junto com os retornos em 30/60/90/180/365 dias pós-sinal. Histogramas, barras comparativas, tabela dos 20 eventos mais recentes.

2. **Múltiplos títulos** — aplica a mesma análise em todos os IPCA+ com 3+ anos de histórico. Agrega os resultados e mostra: quantos títulos se beneficiaram? Qual o excesso médio vs baseline em cada horizonte?

3. **Grid search de parâmetros** — varre combinações de `(janela, threshold)` e gera um heatmap visual mostrando onde a estratégia é mais robusta. **Este é o output mais importante**: zonas grandes de verde = parâmetros robustos; pontos verdes isolados em mar vermelho = overfitting.

## 🧠 O que aprendemos com o teste sintético

Rodando com dados sintéticos que simulam pânicos reais (Covid 2020, Ucrânia 2022), o resultado foi revelador:

- **Com janela de 60 pregões (padrão do notebook original)**: sinal ruim. Excesso negativo em quase todos os horizontes.
- **Com janela de 90-120 pregões**: sinal **muito bom** — 100% dos títulos se beneficiam, excesso de +3% a +5% em 90 dias, hit rate de 60-66%.

**Moral**: janela curta demais capta ruído; janela média capta o regime. O grid search vai te mostrar isso nos seus dados reais.

## ⚠️ Limitações

- Retornos **brutos** (sem custódia B3 e sem IR) — o líquido é menor, especialmente em horizontes curtos.
- Comparação de **venda** é apenas diagnóstica (não é possível shortear TD como PF).
- O que importa pra quem segura IPCA+ até o vencimento é **carrego**, não PU intermediário. A análise é útil para quem tenta fazer market timing de entrada.

## 🔁 Próximos experimentos que você pode fazer

1. Trocar o título analisado (célula com `TITULO_ALVO`).
2. Mudar `WINDOW` e `ABS_THRESHOLD` para rodar com os parâmetros que o heatmap apontou como melhores.
3. Adicionar filtro de regime macro (ex: só valer quando Selic está acima da média).
4. Combinar sinal Fat Tail com J4/Z do Scanner — só comprar quando AMBOS disparam.
