# Equidade na Detecção de Fraudes Financeiras: Impacto do Viés e Mitigação

Este projeto investiga como o viés estrutural em bases de dados financeiras afeta a equidade dos modelos de detecção de fraudes. O trabalho é baseado no estudo **"Ignorance and Prejudice" in Software Fairness** de Zhang & Harman (2021), analisando os conceitos de **Ignorância** (conjunto de features insuficiente) e **Preconceito** (viés embutido nos dados históricos) através da aplicação do algoritmo **Prejudice Remover**.

---

## 📈 Resultados e Análises

Os experimentos focaram na generalização de viés, treinando modelos em bases com diferentes níveis de disparidade e avaliando-os em cenários distintos.

* **Relação entre Equidade e Desempenho:** O aumento da penalidade (η) reduz a diferença de paridade estatística (|SPD|), mas afeta o F1-Score. O equilíbrio mais eficiente foi encontrado no intervalo de η entre 5 e 15, onde o viés diminui consideravelmente sem causar uma redução drástica do recall, que é essencial para identificar fraudes reais.
* **Generalização do Viés:** Os resultados demonstram que o viés nas predições é fortemente influenciado pela distribuição da base de teste. Modelos avaliados em séries diferentes daquelas em que foram treinados apresentam uma disparidade consistentemente maior, evidenciando que o viés aprendido generaliza de forma assimétrica.
* **Volume e Features:** O aumento do volume de dados não reduziu o viés original; pelo contrário, o modelo reproduziu com maior fidelidade os padrões desiguais. Já o uso de features normalizadas (z-scores por ramo) permitiu que o modelo medisse anomalias individuais, mitigando parte do efeito discriminatório do grupo sensível.

---

## 🛠️ Features Utilizadas

As variáveis foram construídas para capturar comportamentos suspeitos enquanto neutralizam a dependência do atributo sensível (**Ramo de Atividade**):

* `final_de_semana`: indica se a transação ocorreu no sábado ou domingo.
* `pct_saldo_gasto`: proporção do valor da transação em relação ao saldo da conta.
* `zscore_valor_no_titular`: nível de anomalia do valor perante o histórico acumulado do titular.
* `n_transacoes_dia_tit`: volume acumulado de transações realizadas pelo titular no mesmo dia.
* `is_cnab220`: indica transações do tipo crédito (CNAB=220).
* `prop_cnab220_tit`: proporção histórica de uso de crédito pelo titular, calculada com *expanding windows* para evitar vazamento de dados.
* `zscore_valor_no_ramo`: anomalia do valor em relação ao comportamento típico do ramo.
* `zscore_n_trans_mes_no_ramo`: anomalia do volume mensal de transações dentro do ramo.
* `zscore_reg_no_ramo`: anomalia da regularidade de uso do cliente normalizada pelo ramo.
* `zscore_valor_total_mes_no_ramo`: anomalia do volume financeiro mensal no ramo.

---

## 📊 Figuras Geradas

O pipeline gera visualizações para fundamentar a análise técnica (salvas no diretório `./resultados/figuras/`):

* `fig1_vies_original_spd` / `fig1b_vies_original_di`: caracterização do Viés Estatístico (|SPD|) e Disparate Impact (DI) originais das bases.
* `fig2_evolucao_recall` / `fig3_evolucao_vies`: impacto do aumento de η no desempenho e na equidade.
* `fig4_relacao_global` / `fig5_impacto_metricas`: visão geral do comportamento do modelo frente às penalidades.
* `fig6_estabilidade_recall` / `fig7_estabilidade_f1`: análise da variação das métricas de desempenho.
* `fig8_antes_vs_depois_spd` / `fig9_antes_vs_depois_di`: comparação da generalização de viés em cenários de avaliação na mesma série versus série distinta.

---

## 📚 Referência

> J. M. Zhang and M. Harman, ""Ignorance and Prejudice" in Software Fairness," *2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)*, Madrid, ES, 2021, pp. 1436-1447, doi: [10.1109/ICSE43902.2021.00129](https://doi.org/10.1109/ICSE43902.2021.00129).
