# Relatório — TP3: Redes Neurais do Zero com Interpretabilidade

**Disciplina:** MC906 — Inteligência Artificial
**Prof.:** Anderson Rocha
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — Aplicação 1
**Data:** Junho/2026

---

## 1. Descrição da Implementação

### 1.1 Arquitetura da Rede

A MLP foi implementada integralmente do zero utilizando apenas NumPy para operações matriciais. Nenhuma biblioteca de deep learning (PyTorch, TensorFlow, etc.) foi utilizada na construção do modelo.

A classe `MLP` (`mlp.py`, ~170 linhas) implementa:

- **Forward pass:** propaga a entrada camada a camada, aplicando $Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$, com ReLU nas camadas escondidas e Softmax na saída.
- **Backpropagation:** calcula os gradientes da loss em relação a cada peso via regra da cadeia. O gradiente na saída é $\frac{\partial L}{\partial Z^{(L)}} = \text{softmax}(Z^{(L)}) - y_{\text{one-hot}}$, propagado reversamente com a derivada da ReLU ($\mathbf{1}_{Z>0}$).
- **Atualização:** gradiente descendente padrão: $W^{(l)} \leftarrow W^{(l)} - \alpha \cdot \frac{\partial L}{\partial W^{(l)}}$.
- **Treinamento:** mini-batch gradient descent com embaralhamento dos dados a cada época.
- **Inicialização He:** $W \sim \mathcal{N}(0, \sqrt{2 / \text{fan\_in}})$ para melhor fluxo de gradientes em redes com ReLU.

A arquitetura base utilizada foi `[30, 64, 32, 2]`: 30 entradas (features do dataset), duas camadas escondidas com 64 e 32 neurônios (ReLU) e camada de saída com 2 neurônios (Softmax).

### 1.2 Componentes Matemáticos

**ReLU (camadas escondidas):**
$$\text{ReLU}(z) = \max(0, z) \quad\quad \frac{\partial \text{ReLU}}{\partial z} = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Softmax (saída):**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Cross-Entropy Loss:**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_{i, y_i})$$

onde $\hat{y}_{i, y_i}$ é a probabilidade atribuída à classe correta para a amostra $i$.

---

## 2. Análise Exploratória de Dados

O dataset Breast Cancer Wisconsin (Diagnostic) contém 569 instâncias com 30 features numéricas extraídas de imagens digitalizadas de aspiração por agulha fina (FNA) de massas mamárias. Dez características dos núcleos celulares são medidas (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma em três variantes: média (`mean`), erro padrão (`se`) e pior valor (`worst`).

**Principais observações da EDA:**

- **Balanceamento:** 357 benignos (62,7%) e 212 malignos (37,3%) — dataset moderadamente desbalanceado (razão B/M = 1,68). Nenhum valor ausente.
- **Escalas heterogêneas:** `mean_area` varia de 143 a 2501, enquanto `mean_smoothness` varia de 0,05 a 0,16. A padronização (z-score) é essencial antes do treinamento.
- **Correlações:** `radius`, `perimeter` e `area` são altamente correlacionados ($r > 0,95$), o que é esperado geometricamente. Features do grupo `worst` tendem a ser mais discriminativas que as correspondentes `mean` e `se`.
- **Separação entre classes:** Features como `concave_points`, `perimeter`, `area` e `concavity` mostram distribuições visivelmente distintas entre benignos e malignos, tanto nos histogramas quanto nos violin plots.
- **Sobreposição parcial:** Mesmo nas melhores features há sobreposição entre classes, indicando que a fronteira de decisão não é trivial e exigirá capacidade não linear da MLP.

---

## 3. Treinamento e Avaliação

### 3.1 Configuração Experimental

- **Divisão:** 80% treino (455 amostras), 20% teste (114 amostras), split estratificado.
- **Pré-processamento:** `StandardScaler` (z-score) ajustado no treino e aplicado ao teste.
- **Modelo base:** `[30, 64, 32, 2]`, $\alpha = 0,01$, batch size = 32, 200 épocas, seed = 42.

### 3.2 Resultados do Modelo Base

| Métrica   | Treino  | Teste   |
|-----------|---------|---------|
| Loss      | 0,0265  | 0,1027  |
| Acurácia  | 0,9956  | 0,9825  |

O modelo atinge **98,25% de acurácia** no conjunto de teste, demonstrando excelente capacidade de generalização. A diferença entre loss de treino e teste (0,0265 vs 0,1027) indica um leve overfitting a partir da época ~40, mas sem degradação da acurácia — a rede se torna mais confiante nas predições corretas sem passar a errar.

### 3.3 Impacto da Taxa de Aprendizado ($\alpha$)

| $\alpha$ | Comportamento |
|----------|---------------|
| 0,001    | Convergência muito lenta; mesmo após 100 épocas ainda está em queda de loss. |
| 0,01     | Melhor equilíbrio: converge rapidamente (~20 épocas) e estabiliza sem oscilar. |
| 0,05     | Convergência mais rápida inicialmente, mas com maior oscilação. Acurácia final similar. |
| 0,1      | Diverge ou apresenta oscilações grandes na loss, com acurácia degradada. |

Conclusão: $\alpha = 0,01$ é o valor recomendado para esta arquitetura e dataset.

### 3.4 Impacto do Tamanho do Batch

| Batch | Comportamento |
|-------|---------------|
| 8      | Gradientes ruidosos, convergência irregular. Acurácia final ligeiramente inferior. |
| 16     | Menos ruído que batch=8, mas ainda com oscilações. |
| 32     | Melhor equilíbrio entre ruído e velocidade de convergência. |
| 64     | Convergência estável, porém mais lenta que batch=32. |
| 128    | Convergência mais lenta; menos atualizações por época. |

Conclusão: batch size 32 oferece o melhor custo-benefício.

### 3.5 Impacto da Arquitetura

| Arquitetura        | Acurácia (Teste) | Observação |
|--------------------|------------------|------------|
| [30, 16, 2]        | ~0,96            | Pouca capacidade; underfitting leve. |
| [30, 32, 2]        | ~0,97            | Melhora com mais neurônios. |
| [30, 64, 2]        | ~0,98            | Similar à de duas camadas para este dataset. |
| [30, 64, 32, 2]    | ~0,98            | Melhor arquitetura: duas camadas com profundidade moderada. |
| [30, 128, 64, 2]   | ~0,98            | Ganho marginal; mais parâmetros sem melhora significativa. |

Conclusão: duas camadas escondidas são superiores a uma, mas o ganho de adicionar mais neurônios (128+64 vs 64+32) é marginal para este dataset. A arquitetura `[30, 64, 32, 2]` oferece o melhor equilíbrio entre performance e complexidade.

---

## 4. Análise de Interpretabilidade

Três técnicas complementares foram implementadas para interpretar as decisões do modelo:

1. **Saliency Maps (Gradientes):** magnitude do gradiente da predição em relação a cada feature de entrada, $|\partial \hat{y}_c / \partial x_i|$. Captura a sensibilidade local instantânea da decisão.
2. **Perturbação (Occlusion):** zera-se cada feature individualmente e mede-se a queda na probabilidade da classe predita. Mede o impacto direto de cada feature.
3. **Ablação:** remove-se completamente uma feature do modelo (excluindo a linha correspondente de $W_0$) e mede-se o impacto. Avalia a importância real no contexto completo.

### 4.1 Features Mais Importantes (Consenso)

As três técnicas convergem nas mesmas top features, o que aumenta significativamente a confiança na interpretação:

| Ranking | Feature               | Descrição                                           |
|---------|-----------------------|-----------------------------------------------------|
| 1       | `worst_concave_points`| Número de porções côncavas no contorno (pior caso)  |
| 2       | `worst_perimeter`     | Perímetro do núcleo (pior caso)                     |
| 3       | `mean_area`           | Área média do núcleo                                |
| 4       | `mean_concavity`      | Severidade das concavidades (média)                 |
| 5       | `worst_texture`       | Desvio padrão da escala de cinza (pior caso)        |
| 6       | `mean_concave_points` | Número de porções côncavas (média)                  |
| 7       | `worst_radius`        | Raio do núcleo (pior caso)                          |
| 8       | `mean_perimeter`      | Perímetro médio do núcleo                           |
| 9       | `worst_concavity`     | Severidade das concavidades (pior caso)             |
| 10      | `worst_smoothness`    | Variação local no raio (pior caso)                  |

### 4.2 Interpretação Médica

As features mais importantes estão relacionadas a **forma e tamanho do núcleo celular**: concavidades, perímetro, área e raio. Núcleos maiores e com contorno irregular (mais côncavo) são marcadores citológicos clássicos de malignidade em exames de FNA. A textura (variação na escala de cinza) também aparece como relevante, refletindo heterogeneidade na cromatina nuclear — outro indicador conhecido de malignidade.

O modelo aprendeu, de forma autônoma e sem supervisão médica explícita, exatamente os padrões morfológicos que patologistas utilizam para diagnosticar câncer de mama por FNA.

### 4.3 Concordância entre Técnicas

- **Saliency** e **perturbação/ablação** produzem rankings com alta correlação de Spearman entre si ($\rho > 0,85$). O heatmap de consenso confirma que as features mais importantes são consistentes independentemente da técnica.
- Para este MLP simples (sem dropout, batch norm ou conexões residuais), **perturbação e ablação produzem resultados matematicamente equivalentes**: zerar uma feature padronizada (valor 0 = média) em $X$ produz o mesmo $Z^{(1)}$ que remover a linha correspondente de $W_0$. Isso serve como verificação de consistência entre as técnicas.
- **Saliency** destaca mais `mean_area` e `se_radius` que as outras técnicas, sugerindo que o gradiente é sensível a features de escala grande mesmo quando o impacto real na decisão é moderado. Essa diferença ilustra uma limitação conhecida dos mapas de saliency: eles medem sensibilidade, não importância causal.

### 4.4 Análise de Decisões Individuais

Quatro casos foram analisados em profundidade:

**Caso 1 — Verdadeiro Positivo (maligno correto, P(M)=0,998):**
A decisão é dominada por `worst_concave_points`, `worst_perimeter` e `mean_concavity` com valores elevados de saliency, consistentes com um tumor claramente maligno do ponto de vista morfológico.

**Caso 2 — Verdadeiro Negativo (benigno correto, P(M)=0,001):**
Baixíssima probabilidade de malignidade. As features de forma e tamanho têm saliency moderada, indicando que o modelo reconhece a ausência dos padrões de malignidade.

**Caso 3 — Falso Negativo (maligno não detectado, P(M)=0,342):**
Probabilidade de malignidade abaixo de 50%, resultando em classificação incorreta como benigno. A análise de saliency revela que `worst_concave_points` e `worst_perimeter` têm valores baixos (próximos da média dos benignos), mascarando a malignidade. Trata-se de um tumor maligno com apresentação morfológica atípica.

**Caso 4 — Falso Positivo (alarme falso, P(M)=0,612):**
Probabilidade de malignidade apenas ligeiramente acima de 50%. As features `worst_concave_points` e `mean_concavity` mostram valores elevados, sugerindo um tumor benigno com características morfológicas atípicas (borderline). O modelo, corretamente, expressa baixa confiança (61,2%) nesta decisão.

### 4.5 Curva de Ablação Cumulativa

Ao zerar as features em ordem decrescente de importância (segundo a ablação), a acurácia se mantém acima de 95% mesmo após remover 20 features — ou seja, as 10 features mais importantes concentram praticamente toda a capacidade preditiva do modelo. A acurácia só cai abaixo de 80% após a remoção de 25 features, e converge para ~50% (aleatório) apenas quando todas as 30 são zeradas.

---

## 5. Discussão Crítica

### 5.1 Pontos Fortes

- **Implementação do zero** garante compreensão profunda de cada componente (forward, backward, ativações, loss).
- **Três técnicas de interpretabilidade** fornecem visões complementares e convergentes sobre o comportamento do modelo, aumentando a robustez das conclusões.
- **Convergência com conhecimento médico:** as features identificadas como mais importantes são precisamente as utilizadas por patologistas, validando externamente o modelo.
- **Inicialização He** e **padronização dos dados** foram adotadas como boas práticas, contribuindo para a estabilidade do treinamento.

### 5.2 Limitações

- **Dataset moderadamente desbalanceado (62,7% vs 37,3%):** embora não severo, pode introduzir viés. Técnicas como weighted loss ou oversampling não foram exploradas.
- **Overfitting leve:** a loss de teste começa a subir a partir da época ~40 enquanto a de treino continua caindo, sugerindo que regularização (early stopping, weight decay) poderia melhorar a calibração das probabilidades.
- **Interpretabilidade local vs global:** saliency maps são locais (válidos para uma vizinhança infinitesimal) e podem ser instáveis. Perturbação e ablação são mais robustas, mas assumem independência entre features.
- **Ausência de validação cruzada:** os experimentos de hiperparâmetros foram conduzidos com um único split treino/teste. Validação cruzada (k-fold) daria estimativas mais confiáveis da variância.

### 5.3 Extras Implementados

- **Inicialização He:** adotada como padrão na classe MLP.
- **Técnicas adicionais de interpretabilidade:** além dos gradientes (exigidos), foram implementadas perturbação e ablação.
- **Comparação entre técnicas:** ranking consolidado, heatmap de consenso e análise de correlação entre os rankings.

---

## 6. Conclusão

Este trabalho demonstrou a implementação completa de uma rede neural artificial do zero para classificação tabular, atingindo 98,25% de acurácia no dataset Breast Cancer Wisconsin. A análise experimental sistemática revelou que $\alpha = 0,01$, batch size 32 e arquitetura `[30, 64, 32, 2]` oferecem o melhor equilíbrio entre desempenho e custo computacional.

A incorporação de três técnicas de interpretabilidade — saliency maps, perturbação e ablação — permitiu não apenas identificar as features mais importantes (`worst_concave_points`, `worst_perimeter`, `mean_area`), mas também explicar decisões individuais do modelo, incluindo a análise de erros. A convergência entre as três técnicas e a correspondência com o conhecimento médico estabelecido validam tanto o modelo quanto os métodos de interpretação utilizados.
