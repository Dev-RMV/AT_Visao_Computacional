# Relatório Técnico de Execução: Exercício 2 - Classificação com OpenCV DNN vs Keras e Pipeline Integrado

**Disciplina:** Visão Computacional / Robótica
**Ambiente:** Python 3.13 | OpenCV 4.14.0 (`opencv-contrib-python`) | TensorFlow 2.21.0 | Keras 3.15.1 | CPU Intel Core i9-14900KF
**Execução de referência:** 23/09/2026, `python main.py --sem-janela` (20 repetições por imagem no benchmark)

---

## 1. Resumo da Execução

* **Item A:** modelo de classificação pré-treinado (**MobileNetV2**, ImageNet) carregado no **OpenCV DNN** a partir do arquivo TFLite (`cv2.dnn.readNetFromTFLite`); 10 imagens Imagenette de 10 categorias distintas processadas com `cv2.dnn.blobFromImage`, forward pass e extração do **top-3** com confiança sobreposta às imagens (`saidas/top3/`); latência, memória e acurácia top-1 comparadas com a mesma rede carregada nativamente no Keras. A acurácia dos dois backends é **medida** contra `entradas/rotulos.csv` com o mesmo critério (igualdade de rótulo após normalização, sinônimos separados por `|`).
* **Item B:** pipeline sequencial das técnicas dos TPs e exercícios anteriores, em `main.py` e no notebook `exercicio02b.ipynb`, que é gerado e **executado de fato** pelo `nbclient` (as saídas gravadas no notebook são as produzidas pelas células):
  1. *Exercício 1:* captura de frame e `cv2.undistort` com a calibração do Exercício 1;
  2. *TP1:* segmentação de ROI por cor HSV (bola azul) com morfologia;
  3. *TP2:* features ORB (`cv2.ORB_create`, 2000 pontos);
  4. *TP3:* detector de pedestres HOG+SVM (`cv2.HOGDescriptor`), a opção escolhida entre "HOG+SVM ou Haar Cascade";
  5. *Exercício 2A:* classificação da ROI com a MobileNetV2 no OpenCV DNN.
  O tempo de cada etapa é impresso em milissegundos e o frame final com todas as anotações é salvo em `saidas/frame_pipeline_integrado.png`.

---

## 2. Item A: Comparativo de Desempenho (OpenCV DNN vs Keras)

### 2.1. Tabela Comparativa de Métricas

Benchmark com 20 repetições sequenciais (batch = 1) por imagem, após 5 inferências de aquecimento. A memória é o RSS do processo medido com `psutil` **depois** do aquecimento (buffers de inferência já alocados); o Keras roda em subprocesso separado para que o runtime do TensorFlow seja medido isoladamente.

| Métrica | OpenCV DNN (C++ / TFLite) | Keras (TensorFlow 2.x) | Razão |
| :--- | :---: | :---: | :---: |
| Latência média por imagem | 5,63 ms | 64,69 ms | 11,5x |
| Latência mínima | 5,10 ms | 60,11 ms | 11,8x |
| Desvio padrão da latência | 0,30 ms | 4,95 ms | 16x |
| Memória adicional (carregar modelo + aquecer) | 51,7 MB | 433,8 MB | 8,4x |
| RSS total do processo | 94,2 MB | 463,8 MB | 4,9x |
| Acurácia top-1 medida (10 imagens rotuladas) | 90,0 % | 90,0 % | igual |

Os dois backends erram a mesma imagem (`n03028079_01`, church → fountain) e acertam as outras 9; as previsões top-1 coincidem nas 10 imagens, o que confirma que a conversão para TFLite não alterou o comportamento da rede.

### 2.2. Classificações Top-3 Obtidas pelo OpenCV DNN nas 10 Categorias

| Imagem | Gabarito | Top-1 | Top-2 | Top-3 |
| :--- | :--- | :--- | :--- | :--- |
| `n01440764_01.JPEG` | tench | **tench** (91,2 %) | reel (2,2 %) | barracouta (0,9 %) |
| `n02102040_01.JPEG` | English_springer | **English_springer** (93,7 %) | Cardigan (2,0 %) | Border_collie (0,7 %) |
| `n02979186_01.JPEG` | cassette_player ou tape_player | **tape_player** (37,3 %) | cassette_player (28,1 %) | CD_player (19,2 %) |
| `n03000684_01.JPEG` | chain_saw | **chain_saw** (42,4 %) | swing (7,2 %) | stretcher (5,9 %) |
| `n03028079_01.JPEG` | church | fountain (52,8 %) - erro | spotlight (3,4 %) | umbrella (1,7 %) |
| `n03394916_01.JPEG` | French_horn | **French_horn** (93,5 %) | trombone (2,2 %) | cornet (1,1 %) |
| `n03417042_01.JPEG` | garbage_truck | **garbage_truck** (92,8 %) | trailer_truck (4,6 %) | moving_van (1,1 %) |
| `n03425413_01.JPEG` | gas_pump | **gas_pump** (86,6 %) | cash_machine (10,6 %) | pay-phone (0,4 %) |
| `n03445777_01.JPEG` | golf_ball | **golf_ball** (18,0 %) | croquet_ball (7,3 %) | soccer_ball (3,3 %) |
| `n03888257_01.JPEG` | parachute | **parachute** (98,2 %) | airship (0,1 %) | balloon (0,1 %) |

---

## 3. Discussão Técnica: Quando o OpenCV DNN é Preferível em Sistemas Embarcados?

1. **Memória:** computadores de bordo (Raspberry Pi 4 de 1 GB ou 2 GB, Jetson Nano, SoCs ARM industriais) dividem a RAM entre sistema operacional, controle e percepção. Nesta medição o processo com OpenCV DNN ficou em 94 MB de RSS e o processo com TensorFlow/Keras em 464 MB. Em placas de 1 GB essa diferença decide se o nó de percepção convive com o resto do sistema ou é encerrado pelo OOM killer.
2. **Latência e determinismo:** o OpenCV DNN executa kernels C++ vetorizados (AVX2/AVX-512 em x86, NEON em ARM) sem o despacho por Python do Keras: 5,6 ms contra 64,7 ms por imagem com batch = 1, com desvio padrão de 0,30 ms contra 4,95 ms. Malhas de controle toleram mal os picos de latência do segundo caso.
3. **Dependências:** para um nó ROS 2 em C++ o OpenCV DNN dispensa interpretador Python e o TensorFlow completo (centenas de MB em disco), bastando o arquivo TFLite ou ONNX.
4. **Quando o Keras é a escolha:** treino, fine-tuning e prototipagem, onde diferenciação automática e o ecossistema de dados importam mais que latência. A prática recomendada é treinar no Keras, exportar para TFLite/ONNX e inferir com OpenCV DNN no dispositivo.

---

## 4. Item B: Pipeline Integrado de Percepção

O pipeline foi executado sobre `entradas/bola.mp4` no **frame 180**. Entre os frames 30 e 75 do vídeo a bola está cortada pela borda superior; a partir do frame 80 ela está inteira no quadro, e no frame 180 aparece grande e completa, o que dá uma ROI e uma classificação com sentido (o frame pode ser trocado com `--frame`).

**Limitação documentada:** `entradas/calibracao.npz` é a calibração da câmera virtual sintética do Exercício 1 (k1 = -0,30, barril forte) e não corresponde à câmera que gravou `bola.mp4`. O passo de undistort é executado e cronometrado porque o enunciado exige encadeá-lo, mas sobre este vídeo ele não corrige distorção real.

### 4.1. Tempos de Execução por Etapa (`main.py`)

| Etapa | Técnica | Referência | Tempo (ms) | Saída |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Undistort com calibração | Exercício 1 | 7,29 | Frame retificado 1280x720 |
| 2 | Segmentação de cor HSV + morfologia | TP1 | 1,79 | ROI (x = 425, y = 247, w = 261, h = 258) |
| 3 | Features ORB (2000 pontos) na ROI | TP2 | 81,43 | 62 keypoints |
| 4 | Detector HOG+SVM de pessoas no frame | TP3 | 77,71 | 0 pedestres (esperado: o vídeo não contém pessoas) |
| 5 | Classificação MobileNetV2 (OpenCV DNN) da ROI | Exercício 2A | 13,63 | croquet_ball 20,1 % / balloon 16,1 % / vase 11,3 % |
| Total | Pipeline completo | | 181,85 | 5,5 FPS em CPU |

Observações:
* ImageNet não tem a classe "bola azul"; o top-3 mostra as classes mais próximas que a MobileNetV2 conhece, com confiança baixa. É o comportamento esperado de um classificador de 1000 classes fixas fora do seu vocabulário.
* O HOG+SVM é executado sobre o frame inteiro e cronometrado; como o vídeo não tem pessoas, 0 detecções é o resultado correto, e a ROI classificada na etapa 5 é a do HSV.
* O notebook executa as mesmas células e grava seus próprios tempos (na execução de referência: 170,0 ms no total, com o mesmo frame, a mesma ROI e o mesmo top-3).

---

## 5. Artefatos Produzidos

* `main.py`: código-fonte completo (Itens A e B, geração e execução do notebook);
* `exercicio02a.py`: script de execução do Item A (`--loops`, `--sem-janela`);
* `exercicio02b.py`: script de execução do Item B (`--frame`, `--sem-janela`), que também gera e executa o notebook;
* `exercicio02b.ipynb`: notebook sequencial com as 5 etapas, executado pelo `nbclient` (saídas e figura reais);
* `saidas/top3/`: 10 imagens anotadas com o top-3 e barras de confiança;
* `saidas/frame_pipeline_integrado.png`: frame final do pipeline com todas as anotações e a telemetria por etapa.
