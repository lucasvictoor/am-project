# Registro de Experimentos — RQs 1–8

- **Data de execução:** 11/03/2026
- **Tempo total:** ~198 segundos
- **Protocolo:** Validação cruzada entre bases — treina em cada uma das 10 bases, testa nas outras 9 (90 pares por condição experimental)

---

## Perguntas de Pesquisa (RQs)

O estudo investiga como características da base de dados e estratégias de modelagem afetam a equidade (*fairness*) de modelos de aprendizado de máquina aplicados à detecção de fraudes em transações financeiras. As nove perguntas de pesquisa são:

| RQ | Pergunta | Experimento |
|----|----------|-------------|
| **RQ1** | A quantidade de atributos (*features*) garante maior justiça no modelo? | Exp. 2 — Feature Sets |
| **RQ2** | Quando temos mais dados de treinamento, podemos ter mais viés? | Exp. 1 — Volume |
| **RQ3** | Aumentar a quantidade de atributos traz precisão semelhante ao aumento de equidade? | Exp. 2 — Feature Sets |
| **RQ4** | Um modelo treinado com quantidade insuficiente de dados pode reforçar preconceitos presentes na base? | Exp. 1 — Volume |
| **RQ5** | O aumento no volume de dados de treinamento pode prejudicar a equidade do modelo? | Exp. 1 — Volume |
| **RQ6** | Um conjunto de dados maior implica necessariamente em maior injustiça? | Exp. 1 — Volume |
| **RQ7** | Balancear os dados tem impacto positivo limitado na equidade quando comparado com outros métodos de mitigação de viés? | Exp. 3 — Mitigação |
| **RQ8** | Métodos de melhoria de equidade, como reponderação de amostras, trazem melhorias significativas na equidade? | Exp. 3 — Mitigação |
| **RQ9** | Técnicas específicas de mitigação de preconceito, como o Prejudice Remover, conseguem melhorar o equilíbrio entre *fairness* e acurácia? | `treino.py` (η sweep) |

> **Nota:** RQ9 é coberta pelo pipeline principal (`treino.py`) com varredura de η ∈ {0, 1, 5, 10, 15, 25, 35, 50}. As RQs 1–8 são investigadas pelos três experimentos descritos neste documento.

### Métricas utilizadas

**Desempenho preditivo**
- **F1-Score** — equilíbrio entre precision e recall; métrica principal de performance
- **Precision** — fração das predições positivas que são corretas
- **Recall** — fração das fraudes reais que o modelo detecta

**Equidade (*Fairness*)**
- **abs_SPD** (*Statistical Parity Difference* absoluta) — |P(ŷ=0 | S=0) − P(ŷ=0 | S=1)|; mede a diferença de taxa de predição favorável entre grupos. Justo se ≤ 0,1.
- **DI** (*Disparate Impact*) — P(ŷ=0 | S=0) / P(ŷ=0 | S=1); razão das taxas de predição favorável. Justo se ≥ 0,8 (regra dos quatro quintos).

Onde S=0 é o grupo não-privilegiado (RAMO_ATIVIDADE_1 = 4, mapeado para 0) e ŷ=0 é a classe favorável (transação aprovada/normal).

---

## Contexto e protocolo experimental

O conjunto de dados é composto por 10 bases de transações financeiras organizadas em duas séries
(**PD** e **GS**), cada uma com 5 níveis de viés sintético introduzido progressivamente
(v = 0,0; 0,25; 0,50; 0,75; 1,0). A variável sensível é `RAMO_ATIVIDADE_1`
(binária: 0 = grupo não-privilegiado, 1 = grupo privilegiado); o alvo é `I-d`
(0 = transação aprovada/normal, classe favorável; 1 = fraude/anomalia).

Duas métricas de justiça são reportadas:

- **abs_SPD** (*Statistical Parity Difference* absoluta): diferença entre a taxa de predição favorável para o grupo não-privilegiado e o privilegiado. Valores menores indicam maior paridade.
- **DI** (*Disparate Impact*): razão entre essas taxas. O ideal é DI = 1,0; valores abaixo de 0,8 caracterizam discriminação pela regra dos quatro quintos.

Todos os experimentos usam `MAX_TESTE = 20 000` amostras por base de teste e `SEED = 42`.

---

## Experimento 1 — Efeito do Volume de Dados de Treino (RQs 2, 4, 5, 6)

**Configuração:** Regressão Logística (LR) baseline com balanceamento por classe, treinada com volumes
N ∈ {1 000, 2 500, 5 000, 7 500, 10 000}, usando as 10 features sem a variável sensível.

### Resultados globais

| Volume | F1     | Precision | Recall | abs_SPD | DI     |
|-------:|--------|-----------|--------|---------|--------|
| 1 000  | 0,9777 | 0,9593    | 0,9972 | 0,3515  | 0,7225 |
| 2 500  | 0,9837 | 0,9702    | 0,9977 | 0,3548  | 0,7206 |
| 5 000  | 0,9882 | 0,9784    | 0,9982 | 0,3569  | 0,7196 |
| 7 500  | 0,9899 | 0,9820    | 0,9981 | 0,3575  | 0,7193 |
| 10 000 | 0,9911 | 0,9844    | 0,9980 | 0,3580  | 0,7190 |

### Análise

**O volume de treino tem impacto marginal sobre a performance e quase nulo sobre a justiça.**
O ganho de F1 entre 1 000 e 10 000 amostras é de apenas **+0,0134** — menos de 1,4% de melhoria
relativa. A precision cresce de forma mais consistente (+0,025), indicando que volume adicional
reduz falsos positivos. O recall, porém, permanece praticamente estável (variação < 0,001),
sugerindo que o modelo já detecta quase todas as fraudes com amostras relativamente pequenas.

No plano da justiça, o resultado é paradoxal: o abs_SPD **aumenta levemente** com o volume
(+0,0065), e o DI **piora** ligeiramente (afasta-se de 1,0). Isso indica que mais dados de treino
não corrigem, e até podem acentuar, o viés aprendido — o modelo simplesmente reproduz com mais
fidelidade os padrões desiguais presentes nos dados, independentemente da quantidade de exemplos.

**Efeito da série:** A série **GS** consistentemente supera **PD** em F1 (≈ +0,005 em todos os
volumes), mas com abs_SPD levemente maior (≈ +0,047). A série PD apresenta F1 ligeiramente mais
baixo porém abs_SPD menor — padrão que se mantém estável independentemente do volume.

**Efeito do nível de viés da base de treino:** Bases com viés intermediário (v = 0,75) produzem
modelos com F1 mais alto (até 0,9958), enquanto bases sem viés (v = 0,00) geram os piores F1
(0,9886 com 10 000 amostras). Esse padrão, contra-intuitivo, se mantém em todos os volumes: treinar
em dados com maior viés sintético gera modelos com melhor discriminação preditiva quando avaliados
nas outras bases, possivelmente porque o viés introduz separabilidade entre classes mais nítida
nos dados de treino. No eixo da justiça, ocorre o inverso: bases com maior viés produzem menor
abs_SPD nas predições (v = 0,00 → 0,3843; v = 0,75 → 0,3461 com 10 000 amostras).

**Conclusão para RQs 2, 4, 5, 6:** O volume de dados a partir de ~1 000 amostras balanceadas
já fornece resultados altamente competitivos (F1 = 0,9777). Investir em coleta massiva de dados,
sem técnicas de mitigação, não resolve o problema de viés e traz retornos decrescentes em
performance. O abs_SPD cresce monotonicamente com o volume, confirmando que mais dados agravam
marginalmente o viés de predição.

---

## Experimento 2 — Efeito do Conjunto de Features (RQs 1, 3)

**Configuração:** LR baseline, N = 10 000 balanceado, 4 conjuntos de features progressivos,
sem variável sensível.

| Feature Set        | Features incluídas                                              | F1     | Precision | Recall | abs_SPD | DI     |
|--------------------|-----------------------------------------------------------------|--------|-----------|--------|---------|--------|
| `transacional_2`   | pct_saldo_gasto, zscore_valor_no_titular                        | 0,5322 | 0,3689    | 0,9698 | 0,1866  | 0,7877 |
| `tipo_temporal_4`  | + is_cnab220, final_de_semana                                   | 0,9701 | 0,9428    | 0,9993 | 0,3548  | 0,7175 |
| `comportamental_6` | + n_transacoes_dia_tit, prop_cnab220_tit                        | 0,9844 | 0,9697    | 0,9997 | 0,3610  | 0,7145 |
| `completo_10`      | + zscore_valor_no_ramo, zscore_n_trans_mes_no_ramo,             | 0,9911 | 0,9844    | 0,9980 | 0,3580  | 0,7190 |
|                    | zscore_reg_no_ramo, zscore_valor_total_mes_no_ramo              |        |           |        |         |        |

### Análise

**O salto mais expressivo ocorre na adição de `is_cnab220` e `final_de_semana`** (de `transacional_2`
para `tipo_temporal_4`): F1 sobe de 0,5322 para 0,9701 — um ganho de **+0,4379**. Esse resultado
reflete o poder discriminativo do tipo de lançamento (CNAB220): transações de crédito concentram
quase toda a fraude no dataset, tornando `is_cnab220` um preditor próximo de determinístico.
Sem essa feature, o modelo opera praticamente às cegas.

O conjunto `transacional_2` tem recall alto (0,9698) mas precision colapsada (0,3689), indicando
que com apenas z-score e percentual de saldo o modelo classifica a maioria das transações como
fraude. Ao adicionar `is_cnab220`, a precision salta para 0,9428 (+0,574) com recall quase
inalterado (+0,029) — o modelo passa a distinguir o tipo de transação antes de julgar o valor.

A transição de `tipo_temporal_4` para `comportamental_6` traz ganho mais modesto em F1 (+0,014)
e precision (+0,027), mas recall praticamente estável (+0,000). As features de comportamento do
titular (volume diário de transações e proporção histórica de CNAB220) acrescentam contexto sobre
o perfil do cliente, refinando a decisão em casos marginais sem alterar significativamente a
capacidade de detecção bruta.

O conjunto `completo_10`, com os z-scores por ramo de atividade, fecha o ciclo com ganho adicional
de F1 (+0,007) e precision (+0,015), mas com leve queda de recall (−0,002). A normalização por
ramo remove efeitos de grupo, melhorando a generalização entre bases com distribuições distintas.

**No eixo da justiça, o padrão é não-monótono.** O conjunto mínimo apresenta o menor abs_SPD
(0,1866) — mas isso é artefato da precision colapsada: quando o modelo classifica quase tudo como
fraude, a diferença entre grupos encolhe por indiferença, não por equidade. A adição de `is_cnab220`
eleva o abs_SPD para 0,3548, porque o modelo passa a diferenciar grupos de forma efetiva. Os
conjuntos maiores oscilam entre 0,3548 e 0,3610, sem tendência clara — mais features não mitiga
o viés estrutural dos dados.

**Efeito da série:** GS e PD seguem o mesmo padrão de salto em `tipo_temporal_4`. Em
`transacional_2`, GS obtém F1 = 0,5356 e PD obtém 0,5287 — diferença mínima. Com o conjunto
completo, GS (0,9936) supera PD (0,9886) em +0,005, a mesma diferença observada no Experimento 1.

**Conclusão para RQs 1, 3:** Features adicionais melhoram F1 mas não reduzem o viés estrutural.
O conjunto `tipo_temporal_4` concentra quase todo o ganho preditivo graças ao poder discriminativo
de `is_cnab220`. Os conjuntos maiores refinam a precision com retorno decrescente. A justiça
permanece praticamente constante a partir de `tipo_temporal_4`, independentemente de quantas
features são adicionadas.

---

## Experimento 3 — Comparação de Métodos de Mitigação de Viés (RQs 7, 8)

**Configuração:** N = 10 000, 4 condições: LR sem mitigação (amostra bruta), LR balanceado, LR com
reponderação (Kamiran & Calders, 2012), e Prejudice Remover com η = 10 (Kamishima et al., 2012).
Todas as condições usam 9 features (conjunto `completo_10` sem `is_cnab220`), pois com essa feature
presente o problema preditivo se torna quasi-trivial — `is_cnab220` separa quase toda a fraude
diretamente, tornando o desbalanceamento de classes irrelevante e eliminando a diferença entre métodos.

### Resultados globais

| Método                    | F1     | Precision | Recall | abs_SPD | DI     |
|---------------------------|--------|-----------|--------|---------|--------|
| `lr_sem_mitigacao`        | **0,9663** | **0,9615** | 0,9718 | 0,3425 | 0,7271 |
| `lr_balanceado`           | 0,9616 | 0,9398    | 0,9850 | 0,3450  | 0,7252 |
| `lr_reponderado`          | 0,9608 | 0,9374    | **0,9861** | 0,3433 | **0,7254** |
| `prejudice_remover_eta10` | 0,8857 | 0,8341    | 0,9599 | **0,3323** | 0,6970 |

### Análise

#### LR sem mitigação — Melhor F1, menor recall

O modelo sem balanceamento apresenta o **maior F1** (0,9663) e a **maior precision** (0,9615),
mas o **menor recall** (0,9718) entre os métodos que efetivamente treinam sem restrição de
equidade. Isso é o comportamento esperado: sem forçar equilíbrio entre classes, o modelo é
mais conservador na predição de fraude — erra menos vezes ao acusar, mas deixa mais fraudes
escaparem.

O abs_SPD (0,3425) é o segundo menor entre todos os métodos, ligeiramente abaixo do LR balanceado
(0,3450). Por nível de viés, o padrão se mantém: o modelo sem balanceamento tem precision
consistentemente acima dos demais, com recall abaixo do balanceado em todos os cenários
(diferença média de −0,013).

#### LR balanceado — Referência robusta, melhor recall entre os LRs

O balanceamento igual por classe **maximiza recall** (0,9850) entre os modelos LR, ao custo de
queda de precision (0,9398) e F1 (0,9616) em relação ao não-balanceado. O abs_SPD (0,3450) é
o mais alto entre os quatro métodos — o balanceamento, por si só, não reduz o viés de predição;
ele redistribui erros de forma mais simétrica entre classes, mas não entre grupos sensíveis.

O LR balanceado é o mais estável: F1 varia de 0,9527 (v = 1,00) a 0,9664 (v = 0,25), sem
comportamentos erráticos entre séries ou níveis de viés.

#### LR reponderado — Melhor equidade entre os LRs, diferença marginal

A reponderação de Kamiran & Calders (2012) redistribui o peso amostral inversamente proporcional
à probabilidade conjunta P(S,Y). O efeito sobre equidade é real mas pequeno: abs_SPD = 0,3433
(−0,0017 em relação ao balanceado) e DI = 0,7254 (+0,0001). F1 cai apenas −0,0007 e recall
sobe +0,0011. Na prática, reponderação e balanceamento produzem modelos quase indistinguíveis —
a diferença de equidade está na terceira casa decimal.

Por série, GS apresenta F1 quase idêntico entre os dois (0,9659 vs 0,9656), e PD mostra queda
leve de 0,9572 para 0,9560. A reponderação tem efeito ligeiramente mais pronunciado em abs_SPD
na série PD (−0,003 vs −0,0004 em GS).

#### Prejudice Remover (η=10) — Única redução real de abs_SPD, custo expressivo em F1

O PR com η = 10 é o único método que reduz abs_SPD de forma perceptível: **0,3323** contra
0,3425–0,3450 dos demais (−0,011 a −0,013). Em compensação, F1 cai para 0,8857 (−0,076 em
relação ao balanceado) e precision cai para 0,8341 (−0,106), com recall também abaixo (0,9599).

O comportamento por nível de viés é não-monótono:

| Viés base treino | F1 (PR) | F1 (LR bal) | abs_SPD (PR) | abs_SPD (LR bal) |
|-----------------:|---------|-------------|--------------|-----------------|
| 0,00             | 0,8972  | 0,9609      | 0,3298       | 0,3721          |
| 0,25             | 0,8965  | 0,9664      | 0,3455       | 0,3552          |
| 0,50             | 0,8311  | 0,9633      | 0,3284       | 0,3444          |
| 0,75             | 0,8508  | 0,9646      | 0,3366       | 0,3330          |
| 1,00             | 0,9527  | 0,9527      | 0,3216       | 0,3216          |

Em v = 1,00 (gs_v1, apenas ramo 4), PR e LR balanceado convergem para resultado idêntico —
fallback para LR puro. Em v = 0,50, a queda de F1 é máxima (−0,132) com a maior redução de
abs_SPD (−0,016). Em v = 0,75, o PR praticamente não reduz abs_SPD em relação ao balanceado
(−0,003) mas ainda perde −0,114 de F1 — pior custo-benefício de todos os cenários.

A série PD responde muito pior ao PR (F1 = 0,8199, precision = 0,7424) do que a GS
(F1 = 0,9514, precision = 0,9258). Isso sugere que a correlação entre variável sensível e alvo
em PD é estruturalmente mais difícil de separar pelo regularizador de equidade.

### Síntese comparativa

| Critério                     | Melhor método            |
|------------------------------|--------------------------|
| Performance (F1)             | `lr_sem_mitigacao`       |
| Recall (detecção de fraude)  | `lr_reponderado`         |
| Precision (menos falsos pos.)| `lr_sem_mitigacao`       |
| Equidade (abs_SPD)           | `prejudice_remover_eta10`|
| Equidade entre LRs (abs_SPD) | `lr_reponderado`         |
| Estabilidade entre bases     | `lr_balanceado`          |
| Custo-benefício geral        | `lr_balanceado`          |

**Conclusão para RQs 7, 8:** Nenhum método elimina o viés de predição — todos produzem DI < 0,8,
violando o limiar dos quatro quintos. O balanceamento melhora recall mas **aumenta** levemente o
abs_SPD em relação ao modelo não balanceado (+0,0025), confirmando que RQ7 tem resposta afirmativa:
balancear tem impacto limitado e até contraproducente na equidade. A reponderação oferece a
melhor equidade entre os modelos LR, mas a diferença (−0,0017 em abs_SPD) é marginal — RQ8
tem resposta negativa: melhorias são mínimas, não significativas. O Prejudice Remover é a única
técnica que reduz abs_SPD de forma perceptível (−0,013), mas ao custo de queda expressiva de F1
(−0,076) e comportamento errático por nível de viés e série.

---

## Conclusões Gerais

| Dimensão                | Achado principal                                                                       |
|-------------------------|----------------------------------------------------------------------------------------|
| **Volume de dados**     | Rendimentos marginais a partir de ~1 000 amostras; volume aumenta levemente o viés    |
| **Features**            | `is_cnab220` concentra quase todo o ganho preditivo; z-scores por ramo refinam precision |
| **Método de mitigação** | LR sem balanceamento domina em F1; reponderação é marginalmente melhor em equidade; PR é o único que reduz abs_SPD perceptivelmente, com custo alto de F1 |
| **Viés estrutural**     | Bases com v = 0,75 produzem melhor F1; abs_SPD cai com viés da base de treino         |
| **Série GS vs PD**      | GS consistentemente superior em F1 (+0,005 no vol., +0,043 no PR); PD com abs_SPD menor |
| **Limiar de equidade**  | Nenhum método atinge DI ≥ 0,8 em média; viés é estrutural e resistente a mitigação    |
