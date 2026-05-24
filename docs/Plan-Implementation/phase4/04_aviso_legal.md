# Etapa 4 — Aviso Legal e Limitações do Modelo

## Objetivo

Adicionar ao `app.py` a seção obrigatória de aviso legal e limitações do modelo, conforme especificado no artigo, garantindo que o texto seja exibido em destaque e que as limitações técnicas estejam documentadas de forma clara para o usuário.

---

## 4.1 Aviso legal obrigatório

O texto abaixo é transcrito literalmente do artigo (seção 3.8) e deve aparecer em destaque na interface.

```python
# ── Aviso legal ───────────────────────────────────────────────────────────────
st.markdown("---")
st.warning(
    "⚠️ **Aviso Legal**\n\n"
    "Esta ferramenta é um apoio analítico baseado em padrões históricos. "
    "O desempenho passado não garante resultados futuros. "
    "**Não constitui recomendação de investimento.**",
    icon="⚠️",
)
```

> O componente `st.warning` exibe o texto com fundo amarelo e ícone, garantindo visibilidade. Conforme o artigo, este aviso deve estar obrigatoriamente presente e em destaque.

---

## 4.2 Seção de limitações técnicas

```python
st.subheader("Limitações do Modelo")

st.markdown("""
O modelo LSTM prevê o preço de fechamento do ITUB4 com base exclusivamente
em **padrões históricos de preço e volume**. Ele não possui acesso a:

- **Eventos exógenos não modelados**: pandemias, mudanças abruptas de política
  monetária, intervenções governamentais ou escândalos corporativos.
  [Zanotto & Hölbig (2026)](https://doi.org/10.17648/sitio-novo-v10n1-1879)
  demonstrou empiricamente que esses fatores elevam significativamente o
  erro de predição.
- **Análise fundamentalista**: demonstrativos financeiros, dividendos, guidance
  de resultados e valuation não são considerados.
- **Sentimento de mercado**: notícias, redes sociais e fluxo institucional
  estão fora do escopo do modelo.
- **Horizonte de previsão**: o modelo produz previsões de **apenas 1 dia útil**
  à frente. Extrapolações para múltiplos dias compõem os erros de forma
  exponencial e não devem ser realizadas.

A escolha do ITUB4 como ativo de estudo mitiga parte das limitações acima
(empresa privada com menor interferência governamental), mas não as elimina.
""")
```

---

## 4.3 Rodapé institucional

```python
# ── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Desenvolvido por Gabriel Sucupira · Henrique Ribeiro · Lucas Zanini · Tiago Teraoka — "
    "Universidade Presbiteriana Mackenzie, 2026.1 · "
    "[Repositório GitHub](https://github.com/LucasZanini096/Artificial-Intelligence-Project)"
)
```

---

## 4.4 Ordem final das seções no `app.py`

A sequência de blocos no arquivo deve seguir esta ordem para garantir UX e conformidade com o artigo:

```
1. st.set_page_config(...)
2. Cabeçalho (título + caption)
3. st.markdown("---")
4. [carregamento silencioso dos artefatos via utils.py]
5. Subheader: "Preço Real vs. Previsão LSTM vs. Baseline"
6. st.plotly_chart(fig, ...)                ← gráfico interativo
7. st.markdown("---")
8. Subheader: "Métricas de Avaliação"
9. st.columns(3) com RMSE / MAE / MAPE     ← painel de métricas
10. Tabela comparativa
11. st.expander("Detalhes do Modelo")
12. st.markdown("---")
13. st.warning(aviso legal)                 ← OBRIGATÓRIO
14. st.subheader("Limitações do Modelo")
15. st.markdown(texto de limitações)
16. st.markdown("---")
17. st.caption(rodapé)
```

---

## 4.5 Teste manual de aceitação

Após executar `streamlit run src/app/app.py`:

| Critério | Verificação |
|---|---|
| Aviso legal visível sem scroll | Usuário vê o `st.warning` na primeira dobra ou no máximo após rolar levemente |
| Texto obrigatório íntegro | Conferir transcrição literal do artigo |
| Limitações listadas | Todos os quatro pontos descritos na seção 4.2 presentes |
| Links funcionais | Rodapé → GitHub abre corretamente |
| Responsividade | Interface funciona em janelas ≥ 1024px de largura |

---

## Checklist da Etapa 4

- [ ] `st.warning` com aviso legal obrigatório exibido em destaque
- [ ] Texto do aviso corresponde literalmente ao especificado no artigo
- [ ] Seção "Limitações do Modelo" com os quatro pontos documentados
- [ ] Rodapé com nomes dos autores e link para o repositório GitHub
- [ ] Ordem das seções no `app.py` conforme definido em 4.4
- [ ] Teste manual de aceitação aprovado (todos os critérios da tabela 4.5)
