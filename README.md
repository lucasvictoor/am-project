# Equidade em Modelos de Aprendizado de Máquina para Detecção de Fraudes: Análise do Impacto do Viés nos Dados e da Técnica Prejudice Remover

Este projeto investiga como o viés estrutural em bases de dados financeiras afeta a equidade dos modelos de detecção de fraudes. Trabalho baseado no estudo de **"Ignorance and Prejudice" in Software Fairness** de Zhang & Harman (2021), analisando os conceitos de **Ignorância** (conjunto de features insuficiente) e **Preconceito** (viés injetado nos rótulos) através da aplicação do algoritmo **Prejudice Remover**.

----------

## 📈 Resultados e Análises

Os experimentos focaram na generalização de viés, treinando modelos em bases com diferentes níveis de disparidade e avaliando-os em cenários distintos.

-   **Relação entre Equidade e Desempenho:** O aumento da penalidade ($\eta$) reduz o **Viés Estatístico (|SPD|)**, mas impacta o **F1-Score**. O equilíbrio mais eficiente foi encontrado no intervalo de **$\eta$ entre 5 e 15**, onde o viés diminui consideravelmente sem causar o colapso do **Recall**, essencial para identificar fraudes reais.
    
-   **Dominância da Base de Teste:** Os resultados demonstram que o viés nas predições é fortemente influenciado pela distribuição da **base de teste**. Mesmo modelos com alta mitigação apresentam dificuldades em manter a equidade quando aplicados em ambientes com viés estrutural severo.
    
-   **Volume e Features:** O aumento do volume de dados não reduziu o viés original; ele tendeu a consolidar padrões discriminatórios. Já o uso de **Z-scores por Ramo** permitiu que o modelo focasse em anomalias individuais, mitigando o efeito do grupo sensível.
    

----------

## 🛠️ Features Utilizadas

As variáveis foram construídas para capturar comportamentos suspeitos enquanto neutralizam a dependência do atributo sensível (**Ramo de Atividade**):

-   **final_de_semana:** indica se a transação ocorreu no sábado ou domingo.
    
-   **pct_saldo_gasto:** proporção do valor da transação em relação ao saldo da conta.
    
-   **zscore_valor_no_titular:** nível de anomalia do valor perante o histórico do titular.
    
-   **n_transacoes_dia_tit:** volume de transações realizadas pelo titular no mesmo dia.
    
-   **is_cnab220:** indica transações do tipo crédito (CNAB=220).
    
-   **prop_cnab220_tit:** proporção histórica de uso de crédito pelo titular, calculada sem vazamento de dados.
    
-   **zscore_valor_no_ramo:** anomalia do valor em relação ao comportamento típico do Ramo.
    
-   **zscore_n_trans_mes_no_ramo:** anomalia do volume mensal de transações dentro do Ramo.
    
-   **zscore_reg_no_ramo:** anomalia da regularidade de uso do cliente normalizada pelo Ramo.
    
-   **zscore_valor_total_mes_no_ramo:** anomalia do volume financeiro mensal no Ramo.
    

----------

## 📊 Figuras Geradas

O pipeline gera visualizações para fundamentar a análise técnica (disponíveis em `./resultados/figuras/`):

-   **fig1_vies_original_spd / fig1b_vies_original_di:** caracterização do Viés Estatístico (|SPD|) e Impacto Disparato (DI) originais das bases.
    
-   **fig2_evolucao_recall / fig3_evolucao_vies:** impacto do aumento de $\eta$ no desempenho e na equidade.
    
-   **fig4_relacao_global / fig5_impacto_metricas:** visão geral do comportamento do modelo frente às penalidades.
    
-   **fig6_estabilidade_recall / fig7_estabilidade_f1:** análise da variação das métricas de desempenho.
    
-   **fig8_antes_vs_depois_spd / fig9_antes_vs_depois_di:** comparação da generalização de viés em cenários de teste na Mesma Série vs. Série Diferente.
    

----------

## 📚 Referência

J. M. Zhang and M. Harman, ""Ignorance and Prejudice" in Software Fairness," _2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)_, Madrid, ES, 2021, pp. 1436-1447, doi: 10.1109/ICSE43902.2021.00129.
