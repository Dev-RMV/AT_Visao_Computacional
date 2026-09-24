# Relatório Técnico de Execução: Exercício 1 - Calibração de Câmera e Realidade Aumentada

**Disciplina:** Visão Computacional / Robótica  
**Ambiente:** Python 3.13 | OpenCV 4.14.0 (`opencv-contrib-python`) | NumPy 2.5.3  
**Status:** Executado e Validado com Sucesso  

---

## 1. Resumo da Execução

O presente trabalho resolve de forma estrita e exaustiva todos os requisitos estipulados em `enunciado.txt`:
* **Item A:** Calibração intrínseca e correção de distorção óptica da câmera virtual a partir de 20 imagens sintéticas de padrão de xadrez (9×6 cantos internos). Obtenção e validação da matriz $K$, dos 5 coeficientes de distorção e dos erros de reprojeção individuais e globais, com geração de painel comparativo *Original vs Corrigida* (`painel_undistort.png`).
* **Item B:** Sobreposição de Realidade Aumentada (AR) em tempo real sobre vídeo gravado do tabuleiro (`video_sintetico.avi`). Estimativa contínua da pose 6DoF ($rvec, tvec$) com `cv2.solvePnP`, projeção com `cv2.projectPoints` de um cubo 3D virtual com aresta de 1 quadrado sobre o canto de origem $(0, 0, 0)$, renderização das arestas com cores distintas por face e exibição de telemetria completa no terminal a cada frame.

---

## 2. Item A: Fundamentação e Parâmetros Físicos da Calibração

### 2.1. Matriz Intrínseca da Câmera ($K$)

A matriz intrínseca $K$ projeta pontos tridimensionais no referencial da câmera para coordenadas em pixels no sensor digital:

$$
K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

* **$f_x, f_y$ (Distâncias Focais em Pixels):**
  * *Significado Físico:* $f_x = f \cdot m_x$ e $f_y = f \cdot m_y$, onde $f$ é a distância focal óptica da lente (em mm) e $m_x, m_y$ são as densidades de pixels do sensor (pixels/mm).
  * *Impacto em Robótica:* Determinam a escala métrica e o Campo de Visão (FOV). Em visão estéreo e odometria visual robótica, $Z = \frac{f_x \cdot B}{d}$ (onde $B$ é a baseline e $d$ a disparidade). Um erro de $1\%$ na distância focal reflete-se em $1\%$ de erro sistemático na profundidade calculada, afetando diretamente a segurança no planejamento de trajetórias de robôs móveis e manipuladores.
* **$c_x, c_y$ (Ponto Principal Óptico em Pixels):**
  * *Significado Físico:* Ponto de interseção do eixo óptico com o plano do sensor. Embora idealmente no centro geométrico ($W/2, H/2$), desalinhamentos mecânicos de montagem geram offsets de dezenas de pixels.
  * *Impacto em Robótica:* Desvios no ponto principal provocam erros sistemáticos de orientação angular no vetor de mira (los - line of sight) do robô, resultando em falhas de preensão (grasping) em braços mecânicos.
* **$s$ (Skew / Cisalhamento):**
  * *Significado Físico:* Ângulo entre os eixos de pixels. É nulo ($0.0$) em sensores digitais CMOS/CCD contemporâneos com matrizes retangulares ortogonais.

### 2.2. Coeficientes de Distorção ($\text{dist} = [k_1, k_2, p_1, p_2, k_3]$)

Modelam desvios ópticos geométricos reais que violam o modelo pinhole ideal (Brown-Conrady):
* **$k_1, k_2, k_3$ (Distorção Radial):**
  * Desvios dependentes da distância radial $r = \sqrt{x^2 + y^2}$ ao centro óptico.
  * $k_1 < 0$ indica distorção tipo **Barril** (*barrel*), clássica de webcams e lentes grande-angulares (linhas periféricas curvadas para fora).
  * Em robótica, se não corrigida, corrompe algoritmos de extração de retas (LSD, Hough) e mapas de ocupação (*occupancy grid*).
* **$p_1, p_2$ (Distorção Tangencial / Descentramento):**
  * Decorrem da falta de paralelismo exato entre o plano das lentes e o plano de silício do sensor.

### 2.3. Erro de Reprojeção Aceitável em Aplicações Robóticas

O Erro Médio de Reprojeção (MRE / RMS) mede a distância euclidiana média (em pixels) entre os cantos detectados no sensor e as projeções matemáticas dos pontos 3D conhecidos:
* **$\text{MRE} < 0.5$ pixels (Excelente / Padrão Ouro):**
  * Mandatório para manipulação robótica de alta precisão (*pick-and-place* milimétrico), calibração mão-olho (*hand-eye*), SLAM visual (ORB-SLAM) e Realidade Aumentada sem trepidação perceptual (*jitter*).
* **$0.5 \le \text{MRE} \le 1.0$ pixel (Bom / Aceitável):**
  * Adequado para navegação autônoma de robôs móveis terrestres em média escala e evasão de obstáculos volumosos.
* **$\text{MRE} > 1.0$ pixel (Inaceitável):**
  * Denota calibração defeituosa, provocando rápido acúmulo de *drift* em odometria visual, erros graves de triangulação 3D e instabilidade severa em AR.

---

## 3. Resultados Quantitativos da Calibração (Item A)

A calibração foi executada utilizando as 20 imagens sintéticas presentes em `entradas/`:

### 3.1. Tabela Comparativa: Calibrado vs Gabarito (`verdade.npz`)

| Parâmetro | Valor Verdadeiro | Valor Calibrado | Erro Absoluto | Erro Relativo (%) |
| :--- | :---: | :---: | :---: | :---: |
| **$f_x$** | 1000.0000 px | 1000.3806 px | +0.3806 px | +0.038% |
| **$f_y$** | 1010.0000 px | 1010.4898 px | +0.4898 px | +0.048% |
| **$c_x$** | 652.0000 px | 652.0135 px | +0.0135 px | +0.002% |
| **$c_y$** | 351.0000 px | 351.2732 px | +0.2732 px | +0.078% |
| **$k_1$** | -0.30000 | -0.30246 | -0.00246 | N/A |
| **$k_2$** | +0.10000 | +0.11252 | +0.01252 | N/A |
| **$p_1$** | +0.00100 | +0.00096 | -0.00004 | N/A |
| **$p_2$** | -0.00060 | -0.00062 | -0.00002 | N/A |
| **$k_3$** | -0.01500 | -0.03407 | -0.01907 | N/A |

### 3.2. Erros de Reprojeção
* **Erro Médio de Reprojeção:** `0.0482 pixels`
* **Erro RMS Global:** `0.0491 pixels`
* **Classificação:** Altamente rigoroso ($< 0.05$ px), correspondendo a menos de um vigésimo de pixel.

---

## 4. Item B: Realidade Aumentada e Rastreamento do Cubo 3D

### 4.1. Arquitetura do Cubo e Cores Distintas por Face
O cubo 3D virtual foi construído com aresta exatamente igual a 1 quadrado ($s = 25$ mm) sobre a origem $(0, 0, 0)$:
* **Base (Vértices 0, 1, 2, 3):** Verde Esmeralda `(0, 200, 0)`
* **Topo (Vértices 4, 5, 6, 7):** Laranja Solar `(0, 140, 255)`
* **Frente (Vértices 0, 1, 5, 4):** Ciano Elétrico `(255, 230, 0)`
* **Direita (Vértices 1, 2, 6, 5):** Vermelho Carmesim `(30, 30, 240)`
* **Fundo (Vértices 2, 3, 7, 6):** Amarelo Ouro `(0, 215, 255)`
* **Esquerda (Vértices 3, 0, 4, 7):** Magenta `(240, 32, 160)`

### 4.2. Estabilidade Temporal
Para garantir que o cubo permaneça rigorosamente estável e aderido ao tabuleiro ao mover a câmera:
1. **Refinamento Subpixel:** `cv2.cornerSubPix` com critério EPS `0.001` e janela adaptativa;
2. **Consistência de Orientação:** Verificação de distância do canto de origem para evitar reversões de 180° que flipariam o cubo;
3. **PnP Contínuo:** `cv2.solvePnP` configurado com `cv2.SOLVEPNP_ITERATIVE` e `useExtrinsicGuess=True` alimentado pela pose anterior, com fallback para `SOLVEPNP_IPPE`;
4. **Métrica de Jitter:** Medição da aceleração espúria dos vértices projetados ($\Delta^2 P$). O jitter médio medido em todo o vídeo foi de **0.1975 pixels/frame²**, confirmando movimentação suave sem vibrações perceptíveis.

---

## 5. Artefatos Produzidos

Os seguintes artefatos foram gerados na pasta `saidas/`:
1. `painel_undistort.png`: Painel comparativo de validação lado a lado (Original vs Corrigida com cv2.undistort);
2. `calibracao.npz`: Arquivo NumPy contendo $K$, distorção, padrão, escala e erros de reprojeção;
3. `video_aumentado.avi`: Vídeo com o cubo 3D e telemetria sobrepostos em todos os 150 frames;
4. `poses_estimadas.csv`: Registro frame a frame de $rvec$, $tvec$, erro de reprojeção e jitter;
5. `frame_cubo_inicial.png`, `frame_cubo_meio.png`, `frame_cubo_final.png`: Capturas de tela para auditoria visual.
