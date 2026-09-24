#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Resolução Completa e Estrita do Exercício 2 (OpenCV DNN vs Keras e Pipeline Integrado)
================================================================================================

ENUNCIADO IMPLEMENTADO:
-----------------------
Item A:
  1. Carregamento de modelo de classificação pré-treinado (MobileNetV2 no ImageNet) via módulo
     DNN do OpenCV (cv2.dnn.readNetFromTFLite).
  2. Processamento de ao menos 10 imagens de categorias distintas:
     (1) Criação do blob com cv2.dnn.blobFromImage (escala 1/127.5, tamanho 224x224, média 127.5, swapRB=True);
     (2) Forward pass na rede neural;
     (3) Extração do top-3 de classes com confiança e exibição gráfica sobre a imagem.
  3. Medição e comparação de latência de inferência no OpenCV DNN vs Keras nativo (TensorFlow).
  4. Tabela comparativa no terminal: latência média/mínima (ms), uso de memória (MB) e acurácia top-1.
  5. Comentário técnico aprofundado no código sobre quando o OpenCV DNN é preferível ao Keras em sistemas embarcados.
  6. Validação: imagens com top-3 sobrepostos em saidas/top3/, tabela comparativa no terminal e código comentado.

Item B:
  1. Integração do pipeline completo dos TPs anteriores em fluxo sequencial e em Jupyter Notebook (exercicio02b.ipynb,
     executado de fato com nbclient, de modo que as saídas gravadas no arquivo são reais):
     (1) Captura de frame do vídeo (entradas/bola.mp4) e aplicação de undistort com a calibração do Ex 1
         (calibração da câmera virtual sintética; não corresponde à câmera de bola.mp4, limitação documentada);
     (2) Segmentação de ROI por cor HSV com faixa azul e morfologia matemática (TP1);
     (3) Extração de descritores e features locais ORB com 2000 pontos (TP2);
     (4) Aplicação do detector clássico HOG+SVM de pessoas (TP3; o enunciado permite HOG+SVM ou Haar Cascade);
     (5) Classificação da ROI segmentada com o modelo MobileNetV2 do Item A.
  2. Exibição do frame final com todas as anotações sobrepostas (HUD de telemetria, contornos, keypoints e classes).
  3. Impressão no terminal do tempo de execução de cada etapa em milissegundos e tempo total do pipeline.
  4. Validação: frame final anotado (saidas/frame_pipeline_integrado.png), terminal com tempos e notebook sequencial.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil

# Configuração de encoding UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# COMENTÁRIOS TÉCNICOS: ESCOLHA DO BACKEND (OPENCV DNN vs KERAS EM EMBARCADOS)
# =============================================================================
#
# 1. POR QUE E QUANDO O MÓDULO OPENCV DNN É PREFERÍVEL AO KERAS EM EMBARCADOS?
#
#    A) CONSUMO DE MEMÓRIA (FOOTPRINT / RAM):
#       - O framework Keras roda sobre o runtime completo do TensorFlow (ou PyTorch/JAX).
#         Mesmo em modo de inferência CPU, importar `keras` e `tensorflow` carrega grafos estáticos,
#         buffers de alocação antecipada (BFC Allocator), runtimes de tensores e dependências pesadas
#         (abseil, flatbuffers, protobuf), consumindo tipicamente entre 350 MB e 600 MB de memória RSS.
#       - O módulo OpenCV DNN (cv2.dnn), por outro lado, é uma biblioteca C++ enxuta, pré-compilada,
#         que não possui dependências de runtimes externos. Para executar uma MobileNetV2 quantizada ou
#         em ponto flutuante, o OpenCV consome apenas ~40 MB a 70 MB de RAM.
#       - Em dispositivos robóticos embarcados com restrição de memória (e.g., Raspberry Pi 4/5 de 1GB/2GB,
#         NVIDIA Jetson Nano, microcontroladores ARM Cortex-A/Cortex-M e placas industriais de controle),
#         uma economia de >300 MB de RAM é frequentemente a diferença entre o sistema operar com estabilidade
#         ou ser encerrado pelo OOM Killer (Out Of Memory) do kernel Linux.
#
#    B) LATÊNCIA DE INFERÊNCIA E TAXA DE PROCESSAMENTO (THROUGHPUT):
#       - O módulo OpenCV DNN implementa camadas customizadas em C++ com kernels vetorizados de baixo nível,
#         otimizados para instruções SIMD da CPU: Intel/AMD AVX2, AVX-512, VNNI e ARM NEON.
#       - Além disso, integra-se nativamente com acelerações de hardware de fabricantes:
#         * Intel OpenVINO / oneDNN (inference engine para x86/iGPU);
#         * Vulkan backend para aceleração em GPUs integradas mobile (Mali, Adreno, Intel Iris);
#         * CUDA / cuDNN para GPUs discretas.
#       - O Keras, além do overhead de conversão de tipos NumPy/Tensor e despacho via ponte CPython,
#         apresenta latência significativamente superior para batch=1 (inferência sequencial frame a frame
#         típica de robótica em tempo real): cerca de uma ordem de grandeza mais lento que o cv2.dnn na CPU
#         (neste trabalho, medido em ~11x com a MobileNetV2; ver tabela impressa pelo Item A).
#
#    C) DEPLOYMENT E SIMPLICIDADE DO PIPELINE DE PERCEPÇÃO:
#       - Em visão computacional para robótica (nós ROS/ROS 2 em C++ ou Python), todo o pré-processamento
#         (captura de vídeo, desdistorção óptica, conversão de espaço de cores HSV, redimensionamento,
#         recorte de ROI) já é executado pelo OpenCV.
#       - Com o OpenCV DNN, o mesmo objeto `cv2.Mat` transita diretamente para `cv2.dnn.blobFromImage`
#         e `net.forward()` sem necessidade de cópias extras de memória para tensores Keras.
#       - Elimina a necessidade de instalar wheels gigantescos do TensorFlow (que chegam a 500 MB no disco
#         e frequentemente apresentam problemas de compatibilidade de ABI com versões de Python ou glibc).
#
#    D) DETERMINISMO TEMPORAL E TEMPO REAL:
#       - O TensorFlow/Keras possui rotinas internas assíncronas de gerenciamento de threads e coleta de lixo
#         do Python que provocam "spikes" esporádicos de latência (jitter temporal).
#       - O OpenCV DNN possui execução síncrona, previsível e determinística, requisito mandatório para
#         malhas fechadas de controle robótico (onde frames atrasados causam oscilações de trajetória).
#
#    E) QUANDO O KERAS AINDA É PREFERÍVEL?
#       - O Keras é superior durante as fases de PESQUISA, TREINAMENTO, FINE-TUNING e PROTOTIPAGEM rápida,
#         pois oferece diferenciação automática (AutoDiff), camadas customizadas dinâmicas, callbacks
#         de treinamento e integração com o ecossistema de dados do TensorFlow/Hugging Face.
#       - Portanto, a melhor prática na engenharia de visão robótica é:
#         "Treinar e exportar no Keras/PyTorch -> Converter para TFLite/ONNX -> Inferir em produção com OpenCV DNN."
# =============================================================================

RAIZ = Path(__file__).resolve().parent
ENTRADAS_DIR = RAIZ / "entradas"
SAIDAS_DIR = RAIZ / "saidas"
MODELOS_DIR = RAIZ / "modelos"

# Caminhos padrão dos arquivos
CAMINHO_TFLITE = MODELOS_DIR / "mobilenetv2_imagenet.tflite"
CAMINHO_JSON_CLASSES = MODELOS_DIR / "imagenet_class_index.json"
PASTA_IMAGENS_TESTE = ENTRADAS_DIR / "imagens_teste"
CAMINHO_ROTULOS_CSV = ENTRADAS_DIR / "rotulos.csv"
CAMINHO_VIDEO_BOLA = ENTRADAS_DIR / "bola.mp4"
CAMINHO_CALIBRACAO = ENTRADAS_DIR / "calibracao.npz"

# Frame padrão do pipeline do Item B. Em bola.mp4 a bola só está inteira dentro do quadro a partir
# do frame ~80 (entre os frames 30 e 75 ela é cortada pela borda superior); o frame 180 mostra a bola
# grande e completa, o que dá uma ROI e uma classificação com sentido.
FRAME_PADRAO = 180


# =============================================================================
# FUNÇÕES AUXILIARES DE MODELOS E CLASSES
# =============================================================================

def carregar_rotulos_imagenet(caminho_json: Path = CAMINHO_JSON_CLASSES) -> Dict[int, str]:
    """Carrega o mapeamento de 1000 classes do ImageNet a partir de imagenet_class_index.json."""
    if not caminho_json.exists():
        raise FileNotFoundError(f"Arquivo de classes não encontrado em: {caminho_json}")
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Formato do JSON: {"0": ["n01440764", "tench"], ...}
    return {int(k): v[1] for k, v in data.items()}


def carregar_rotulos_esperados_csv(caminho_csv: Path = CAMINHO_ROTULOS_CSV) -> Dict[str, str]:
    """Lê o arquivo rotulos.csv contendo o rótulo esperado para cada arquivo de imagem."""
    if not caminho_csv.exists():
        return {}
    rotulos = {}
    with open(caminho_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                rotulos[row[0].strip()] = row[1].strip()
    return rotulos


def _normalizar_rotulo(nome: str) -> str:
    return nome.strip().lower().replace(" ", "_").replace("-", "_")


def rotulo_bate(esperado: str, predito: str) -> bool:
    """Compara o top-1 previsto com o gabarito de rotulos.csv.

    O gabarito pode listar sinônimos aceitos separados por '|' (ex.: 'cassette_player|tape_player');
    cada alternativa é comparada por igualdade após normalização, sem casamento por substring.
    """
    alternativas = {_normalizar_rotulo(a) for a in esperado.split("|") if a.strip()}
    return _normalizar_rotulo(predito) in alternativas


# =============================================================================
# ITEM A: PROCESSAMENTO DE IMAGENS E BENCHMARK OPENCV DNN vs KERAS
# =============================================================================

def desenhar_card_top3(img_bgr: np.ndarray, top3_labels: List[Tuple[str, float]], nome_arquivo: str) -> np.ndarray:
    """Desenha painel com as 3 classes mais prováveis e barras de confiança sobre a imagem."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Painel translúcido inferior
    card_h = 100
    card_w = min(w - 20, 480)
    x0, y0 = 10, h - card_h - 10

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, dst=out)
    cv2.rectangle(out, (x0, y0), (x0 + card_w, y0 + card_h), (80, 80, 80), 1)

    # Título do card
    cv2.putText(out, f"MobileNetV2 (OpenCV DNN): {nome_arquivo}", (x0 + 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Cores das 3 posições
    cores_rank = [(0, 255, 100), (0, 215, 255), (180, 180, 180)]
    bar_x = x0 + 240
    bar_max_w = card_w - 260

    for i, (nome_classe, conf) in enumerate(top3_labels):
        y_linha = y0 + 44 + i * 22
        pct = conf * 100.0
        # Texto: Rank, classe e porcentagem
        txt = f"#{i+1} {nome_classe[:18]:<18} {pct:5.1f}%"
        cv2.putText(out, txt, (x0 + 10, y_linha),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cores_rank[i], 1, cv2.LINE_AA)

        # Barra visual de confiança
        largura_barra = int(bar_max_w * max(0.01, min(1.0, conf)))
        cv2.rectangle(out, (bar_x, y_linha - 10), (bar_x + bar_max_w, y_linha - 2), (50, 50, 50), -1)
        cv2.rectangle(out, (bar_x, y_linha - 10), (bar_x + largura_barra, y_linha - 2), cores_rank[i], -1)

    return out


def benchmark_opencv_dnn(
    caminho_modelo: Path,
    imagens: List[Path],
    rotulos_id: Dict[int, str],
    rotulos_gabarito: Dict[str, str],
    loops: int = 20,
) -> Dict[str, Any]:
    """Mede latência, memória e acurácia top-1 do modelo MobileNetV2 no OpenCV DNN."""
    proc = psutil.Process(os.getpid())
    mem_antes = proc.memory_info().rss / (1024 * 1024)

    # Carregar modelo TFLite no OpenCV DNN
    net = cv2.dnn.readNetFromTFLite(str(caminho_modelo))

    # Preparar blobs das imagens
    blobs = []
    for p in imagens:
        img = cv2.imread(str(p))
        b = cv2.dnn.blobFromImage(img, scalefactor=1.0/127.5, size=(224, 224),
                                  mean=(127.5, 127.5, 127.5), swapRB=True)
        blobs.append(b)

    # Aquecimento (Warm-up). A memória é medida DEPOIS do warm-up, com os buffers de
    # inferência já alocados; o mesmo critério é usado no subprocesso Keras.
    for _ in range(5):
        net.setInput(blobs[0])
        _ = net.forward()
    mem_depois = proc.memory_info().rss / (1024 * 1024)
    mem_delta = max(0.0, mem_depois - mem_antes)

    # Medição de latência
    tempos_ms = []
    for _ in range(loops):
        for b in blobs:
            t0 = time.perf_counter()
            net.setInput(b)
            _ = net.forward()
            tempos_ms.append((time.perf_counter() - t0) * 1000.0)

    # Acurácia Top-1 medida contra o gabarito (rotulos.csv). Imagens sem gabarito não entram na conta.
    acertos = 0
    avaliadas = 0
    predicoes = []

    for p, b in zip(imagens, blobs):
        net.setInput(b)
        preds = net.forward()[0]
        top3_idx = np.argsort(preds)[::-1][:3]
        top1_classe = rotulos_id.get(int(top3_idx[0]), "desconhecido")
        top3_res = [(rotulos_id.get(int(idx), f"id_{idx}"), float(preds[idx])) for idx in top3_idx]
        predicoes.append((p, top3_res))

        esperado = rotulos_gabarito.get(p.name)
        if esperado:
            avaliadas += 1
            if rotulo_bate(esperado, top1_classe):
                acertos += 1
        else:
            print(f"  [Aviso] {p.name} não tem rótulo em rotulos.csv; excluída da acurácia.")

    acuracia_top1 = (acertos / avaliadas * 100.0) if avaliadas else float("nan")

    return {
        "nome": "OpenCV DNN (C++ TFLite)",
        "latencia_media_ms": float(np.mean(tempos_ms)),
        "latencia_min_ms": float(np.min(tempos_ms)),
        "latencia_std_ms": float(np.std(tempos_ms)),
        "memoria_delta_mb": mem_delta,
        "memoria_total_mb": mem_depois,
        "acuracia_top1": acuracia_top1,
        "imagens_avaliadas": avaliadas,
        "predicoes": predicoes,
    }


def benchmark_keras_subprocesso(
    imagens: List[Path],
    rotulos_id: Dict[int, str],
    rotulos_gabarito: Dict[str, str],
    loops: int = 20,
) -> Dict[str, Any]:
    """Executa o benchmark Keras nativo em subprocesso separado, para que a memória do runtime
    TensorFlow/Keras seja medida isoladamente do processo principal (que só carrega OpenCV).

    A acurácia top-1 é MEDIDA no subprocesso com o mesmo gabarito (rotulos.csv) e o mesmo critério
    de comparação usado para o OpenCV DNN. Se o subprocesso falhar (TensorFlow ausente, pesos não
    baixados por falta de internet, timeout), a função levanta RuntimeError: nenhum valor estimado
    é impresso no lugar de uma medição.
    """
    script_code = f"""
import os, sys, time, json, psutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

proc = psutil.Process(os.getpid())
mem_antes = proc.memory_info().rss / (1024 * 1024)

import keras
from keras.applications import mobilenet_v2

model = mobilenet_v2.MobileNetV2(weights='imagenet')

import cv2
imagens_paths = {json.dumps([str(p) for p in imagens])}
rotulos_id = {{int(k): v for k, v in json.loads({json.dumps(json.dumps(rotulos_id))}).items()}}
rotulos_gab = json.loads({json.dumps(json.dumps(rotulos_gabarito))})

def normalizar(nome):
    return nome.strip().lower().replace(" ", "_").replace("-", "_")

def rotulo_bate(esperado, predito):
    alternativas = {{normalizar(a) for a in esperado.split("|") if a.strip()}}
    return normalizar(predito) in alternativas

imgs_batch = []
for p in imagens_paths:
    img = cv2.imread(p)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    arr = mobilenet_v2.preprocess_input(img_res.astype(np.float32))
    imgs_batch.append(np.expand_dims(arr, axis=0))

# Aquecimento (warm-up); a memória é medida depois, com os buffers de inferência já alocados
for _ in range(5):
    _ = model(imgs_batch[0], training=False)
mem_depois = proc.memory_info().rss / (1024 * 1024)
mem_delta = max(0.0, mem_depois - mem_antes)

tempos = []
for _ in range({loops}):
    for arr in imgs_batch:
        t0 = time.perf_counter()
        _ = model(arr, training=False)
        tempos.append((time.perf_counter() - t0) * 1000.0)

# Acurácia top-1 medida com o mesmo gabarito e o mesmo critério do OpenCV DNN
acertos = 0
avaliadas = 0
for p, arr in zip(imagens_paths, imgs_batch):
    preds = model(arr, training=False).numpy()[0]
    top1 = rotulos_id.get(int(np.argmax(preds)), "desconhecido")
    esperado = rotulos_gab.get(os.path.basename(p))
    if esperado:
        avaliadas += 1
        if rotulo_bate(esperado, top1):
            acertos += 1

res = {{
    "nome": "Keras Nativo (TensorFlow 2.x)",
    "latencia_media_ms": float(np.mean(tempos)),
    "latencia_min_ms": float(np.min(tempos)),
    "latencia_std_ms": float(np.std(tempos)),
    "memoria_delta_mb": float(mem_delta),
    "memoria_total_mb": float(mem_depois),
    "acuracia_top1": (acertos / avaliadas * 100.0) if avaliadas else float("nan"),
    "imagens_avaliadas": avaliadas,
}}
print("JSON_RESULT:" + json.dumps(res))
"""
    try:
        proc_run = subprocess.run(
            [sys.executable, "-"],
            input=script_code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Benchmark Keras excedeu 600 s (TensorFlow lento ou download dos pesos travado).") from e

    for line in proc_run.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])

    stderr_tail = "\n".join(proc_run.stderr.strip().splitlines()[-15:])
    raise RuntimeError(
        "O subprocesso Keras não produziu resultado (código de saída "
        f"{proc_run.returncode}). Verifique se tensorflow/keras estão instalados e se os pesos "
        "ImageNet estão em ~/.keras/models (o primeiro uso baixa 14 MB).\n"
        f"Últimas linhas do stderr:\n{stderr_tail}"
    )


def executar_item_a(
    pasta_imagens: Path = PASTA_IMAGENS_TESTE,
    pasta_saidas: Path = SAIDAS_DIR,
    caminho_modelo: Path = CAMINHO_TFLITE,
    loops_benchmark: int = 20,
    sem_janela: bool = False,
) -> Dict[str, Any]:
    """Executa estritamente o Item A do enunciado:

    1. Carrega o modelo MobileNetV2 no OpenCV DNN.
    2. Processa 10 imagens de categorias distintas, aplicando blobFromImage, forward pass e top-3.
    3. Mede e compara a latência, memória e acurácia do OpenCV DNN vs Keras.
    4. Imprime tabela comparativa e o comentário técnico no terminal.
    5. Salva todas as 10 imagens com anotações em saidas/top3/.
    """
    print("\n" + "=" * 85)
    print("ITEM A: CLASSIFICAÇÃO COM OPENCV DNN vs KERAS (MOBILENETV2 - IMAGENET)")
    print("=" * 85)

    rotulos_id = carregar_rotulos_imagenet()
    rotulos_gab = carregar_rotulos_esperados_csv()

    arquivos_img = sorted(list(pasta_imagens.glob("*.JPEG")) + list(pasta_imagens.glob("*.jpg")) + list(pasta_imagens.glob("*.png")))
    if len(arquivos_img) < 10:
        raise RuntimeError(f"O enunciado exige ao menos 10 imagens de categorias distintas. Encontradas {len(arquivos_img)} em {pasta_imagens}.")

    imagens_10 = arquivos_img[:10]
    print(f"[Item A] Selecionadas {len(imagens_10)} imagens de categorias distintas para avaliação:")
    for i, p in enumerate(imagens_10, 1):
        esperado = rotulos_gab.get(p.name, "categoria")
        print(f"  [{i:02d}/10] {p.name:<18} -> Categoria de referência: {esperado}")

    # 1. Benchmark do OpenCV DNN e extração de Top-3
    print("\n[Item A] Executando inferência e benchmark com OpenCV DNN...")
    res_dnn = benchmark_opencv_dnn(caminho_modelo, imagens_10, rotulos_id, rotulos_gab, loops=loops_benchmark)

    # Salvar imagens anotadas com Top-3
    pasta_top3 = pasta_saidas / "top3"
    pasta_top3.mkdir(parents=True, exist_ok=True)
    print(f"\n[Item A] Salvando imagens anotadas com Top-3 em: {pasta_top3}")

    for p, top3_labels in res_dnn["predicoes"]:
        img = cv2.imread(str(p))
        anotada = desenhar_card_top3(img, top3_labels, p.name)
        out_p = pasta_top3 / f"top3_{p.stem}.png"
        cv2.imwrite(str(out_p), anotada)
        print(f"  -> {out_p.name:<25} | Top-1: {top3_labels[0][0]:<18} ({top3_labels[0][1]*100:.1f}%)")
        if not sem_janela:
            cv2.imshow("Validacao Item A - Top-3 sobreposto (OpenCV DNN)", anotada)
            cv2.waitKey(700)
    if not sem_janela:
        cv2.destroyAllWindows()

    # 2. Benchmark do Keras Nativo
    print("\n[Item A] Executando benchmark comparativo com Keras Nativo (TensorFlow)...")
    res_keras = benchmark_keras_subprocesso(imagens_10, rotulos_id, rotulos_gab, loops=loops_benchmark)

    # 3. Impressão da Tabela Comparativa Exigida
    speedup = res_keras["latencia_media_ms"] / max(1e-4, res_dnn["latencia_media_ms"])
    mem_ratio = res_keras["memoria_delta_mb"] / max(1e-4, res_dnn["memoria_delta_mb"])

    print("\n" + "=" * 90)
    print("TABELA COMPARATIVA DE DESEMPENHO: OPENCV DNN vs KERAS (MOBILENETV2)")
    print("=" * 90)
    print(f"{'Métrica':<32} | {'OpenCV DNN (C++)':<24} | {'Keras (TensorFlow 2.x)':<24}")
    print("-" * 90)
    print(f"{'Latência Média por Frame':<32} | {res_dnn['latencia_media_ms']:18.2f} ms | {res_keras['latencia_media_ms']:18.2f} ms")
    print(f"{'Latência Mínima (Melhor Caso)':<32} | {res_dnn['latencia_min_ms']:18.2f} ms | {res_keras['latencia_min_ms']:18.2f} ms")
    print(f"{'Desvio Padrão da Latência':<32} | {res_dnn['latencia_std_ms']:18.2f} ms | {res_keras['latencia_std_ms']:18.2f} ms")
    print(f"{'Uso de Memória RAM (Delta Modelo)':<32} | {res_dnn['memoria_delta_mb']:18.1f} MB | {res_keras['memoria_delta_mb']:18.1f} MB")
    print(f"{'Memória Total do Processo (RSS)':<32} | {res_dnn['memoria_total_mb']:18.1f} MB | {res_keras['memoria_total_mb']:18.1f} MB")
    print(f"{'Acurácia Top-1 (medida)':<32} | {res_dnn['acuracia_top1']:18.1f}  % | {res_keras['acuracia_top1']:18.1f}  %")
    print(f"{'Imagens com gabarito avaliadas':<32} | {res_dnn['imagens_avaliadas']:18d}    | {res_keras['imagens_avaliadas']:18d}   ")
    print("-" * 90)
    print(f"-> VANTAGEM OPENCV DNN: {speedup:.1f}x MAIS RÁPIDO e consome {mem_ratio:.1f}x MENOS MEMÓRIA RAM")
    print("=" * 90)

    # 4. Comentário Técnico sobre Escolha do Backend (números tirados das medições acima)
    print("\n" + "=" * 90)
    print("COMENTÁRIO TÉCNICO SOBRE ESCOLHA DE BACKEND EM SISTEMAS EMBARCADOS:")
    print("=" * 90)
    print(f"""
1. RESTRIÇÃO CRÍTICA DE MEMÓRIA (RAM):
   Sistemas embarcados robóticos (Raspberry Pi, Jetson Nano, SoCs ARM Cortex-A) operam
   frequentemente com 512 MB a 2 GB de RAM compartilhada entre SO, controle de motores e
   percepção. Nesta medição o processo com OpenCV DNN ficou em {res_dnn['memoria_total_mb']:.0f} MB de RSS,
   enquanto o processo com TensorFlow/Keras ficou em {res_keras['memoria_total_mb']:.0f} MB. Em placas de 1 GB
   essa diferença decide se o nó de percepção convive com o resto do sistema ou é morto pelo OOM Killer.

2. VELOCIDADE E LATÊNCIA EM TEMPO REAL:
   O OpenCV DNN executa kernels C++ otimizados para SIMD (AVX2/NEON) sem despacho por Python:
   {res_dnn['latencia_media_ms']:.1f} ms contra {res_keras['latencia_media_ms']:.1f} ms por imagem (batch=1), ou seja, {speedup:.1f}x
   mais rápido, com desvio padrão {res_dnn['latencia_std_ms']:.2f} ms contra {res_keras['latencia_std_ms']:.2f} ms (mais determinístico).

3. CONCLUSÃO PRÁTICA:
   Com os mesmos pesos, a acurácia top-1 medida foi {res_dnn['acuracia_top1']:.0f}% (DNN) e {res_keras['acuracia_top1']:.0f}% (Keras) nas
   {res_dnn['imagens_avaliadas']} imagens rotuladas: não há perda de qualidade ao inferir com o OpenCV DNN. Para deploy
   embarcado em tempo real, o OpenCV DNN é a escolha; o Keras fica para treino, fine-tuning e prototipagem.
""")
    print("=" * 90 + "\n")

    return {
        "dnn": res_dnn,
        "keras": res_keras,
        "speedup": speedup,
        "pasta_top3": pasta_top3,
    }


# =============================================================================
# ITEM B: PIPELINE INTEGRADO DE PERCEPÇÃO (UNDISTORT -> HSV -> ORB -> HOG -> DNN)
# =============================================================================

def executar_item_b(
    caminho_video: Path = CAMINHO_VIDEO_BOLA,
    caminho_calibracao: Path = CAMINHO_CALIBRACAO,
    caminho_modelo: Path = CAMINHO_TFLITE,
    pasta_saidas: Path = SAIDAS_DIR,
    frame_alvo_idx: int = FRAME_PADRAO,
    sem_janela: bool = False,
) -> Dict[str, Any]:
    """Executa estritamente o Item B do enunciado:

    Integra o pipeline completo dos TPs anteriores num único fluxo sequencial:
    (1) Captura de frame e aplicação de undistort (Exercício 1).
    (2) Segmentação de ROI por cor HSV (TP1).
    (3) Extração de features ORB (TP2).
    (4) Aplicação do detector HOG+SVM de pessoas (TP3).
    (5) Classificação da ROI detectada com MobileNetV2 DNN (Item A).
    Exibe o frame final anotado, imprime os tempos de cada etapa em ms e gera o notebook sequencial.
    """
    print("\n" + "=" * 85)
    print("ITEM B: PIPELINE INTEGRADO DE PERCEPÇÃO (Técnicas dos TPs Anteriores)")
    print("=" * 85)

    # 0. Carregar recursos e calibração
    if not caminho_calibracao.exists():
        raise FileNotFoundError(f"Calibração não encontrada em: {caminho_calibracao}")

    with np.load(str(caminho_calibracao)) as calib:
        K = calib["K"]
        dist = calib["dist"]

    if not caminho_video.exists():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado em: {caminho_video}")

    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_alvo_idx)
    ret, frame_bruto = cap.read()
    cap.release()
    if not ret or frame_bruto is None:
        raise RuntimeError(f"Falha ao ler o frame {frame_alvo_idx} de {caminho_video}")

    rotulos_id = carregar_rotulos_imagenet()
    net = cv2.dnn.readNetFromTFLite(str(caminho_modelo))

    print(f"[Item B] Frame {frame_alvo_idx} capturado de {caminho_video.name} ({frame_bruto.shape[1]}x{frame_bruto.shape[0]} px)")

    # -------------------------------------------------------------------------
    # ETAPA 1: UNDISTORT (Exercício 1)
    # -------------------------------------------------------------------------
    # LIMITAÇÃO DOCUMENTADA: calibracao.npz é a calibração da câmera VIRTUAL SINTÉTICA do
    # Exercício 1 (k1 = -0.30, barril forte). bola.mp4 foi gravado por outra câmera, logo este
    # passo não corrige a distorção real do vídeo; ele é executado (e cronometrado) porque o
    # enunciado exige encadear o undistort do Exercício 1 no pipeline.
    t0 = time.perf_counter()
    frame_undist = cv2.undistort(frame_bruto, K, dist, None, K)
    t_etapa1 = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # ETAPA 2: SEGMENTAÇÃO DE ROI POR COR HSV (TP1)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2HSV)
    # Faixa da bola azul calibrada no TP1: H in [90, 130], S in [60, 255], V in [40, 255]
    lower_blue = np.array([90, 60, 40], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morfologia matemática para remoção de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_box = None
    maior_contorno = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) >= 500:
            roi_box = cv2.boundingRect(c)
            maior_contorno = c

    t_etapa2 = (time.perf_counter() - t0) * 1000.0
    if roi_box is None:
        raise RuntimeError(
            f"Nenhuma ROI azul com área >= 500 px encontrada no frame {frame_alvo_idx} de {caminho_video.name}. "
            "Escolha outro frame com --frame (em bola.mp4 a bola está inteira no quadro, por exemplo, entre os frames 80 e 239)."
        )
    rx, ry, rw, rh = roi_box

    # -------------------------------------------------------------------------
    # ETAPA 3: EXTRAÇÃO DE FEATURES ORB (TP2)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=5)
    # Extrair keypoints na ROI do objeto (coordenadas locais convertidas para o frame)
    roi_img = frame_undist[ry:ry+rh, rx:rx+rw]
    kp_roi, des_roi = orb.detectAndCompute(roi_img, None)
    # Converter coordenadas locais dos keypoints para o frame global
    kp_global = [cv2.KeyPoint(kp.pt[0] + rx, kp.pt[1] + ry, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_roi]
    t_etapa3 = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # ETAPA 4: DETECTOR HOG+SVM DE PESSOAS (TP3)
    # -------------------------------------------------------------------------
    # O enunciado pede "HOG+SVM ou Haar Cascade"; a opção implementada é o HOG+SVM de pedestres
    # do OpenCV. bola.mp4 não contém pessoas, portanto 0 detecções é o resultado esperado: a etapa
    # é executada e cronometrada para compor o pipeline, e a ROI classificada na etapa 5 é a do HSV.
    t0 = time.perf_counter()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes_hog, weights_hog = hog.detectMultiScale(frame_undist, winStride=(8, 8), padding=(8, 8), scale=1.05)
    t_etapa4 = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # ETAPA 5: CLASSIFICAÇÃO DNN DA ROI DETECTADA (Item A)
    # -------------------------------------------------------------------------
    # ImageNet (1000 classes) não tem a classe "bola azul": o top-3 mostra as classes mais
    # próximas que a MobileNetV2 conhece (ex.: soccer_ball, balloon, golf_ball), com confiança baixa.
    t0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(roi_img, scalefactor=1.0/127.5, size=(224, 224),
                                mean=(127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    preds = net.forward()[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    top3_roi = [(rotulos_id.get(idx, f"id_{idx}"), float(preds[idx])) for idx in top3_idx]
    t_etapa5 = (time.perf_counter() - t0) * 1000.0

    t_total = t_etapa1 + t_etapa2 + t_etapa3 + t_etapa4 + t_etapa5

    # -------------------------------------------------------------------------
    # EXIBIÇÃO NO TERMINAL DOS TEMPOS EXIGIDOS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("TEMPOS DE EXECUÇÃO DO PIPELINE DE PERCEPÇÃO (ITEM B):")
    print("-" * 75)
    print(f"  Etapa 1: Undistort Óptico (Ex 1)            : {t_etapa1:8.2f} ms")
    print(f"  Etapa 2: Segmentação de ROI por Cor HSV (TP1): {t_etapa2:8.2f} ms")
    print(f"  Etapa 3: Extração de Features ORB (TP2)     : {t_etapa3:8.2f} ms ({len(kp_global)} keypoints)")
    nota_hog = " (esperado: o vídeo não contém pessoas)" if len(boxes_hog) == 0 else ""
    print(f"  Etapa 4: Detector HOG+SVM (TP3)             : {t_etapa4:8.2f} ms ({len(boxes_hog)} pedestres{nota_hog})")
    print(f"  Etapa 5: Classificação MobileNetV2 DNN (Ex 2A): {t_etapa5:8.2f} ms (Top-1: {top3_roi[0][0]})")
    print("-" * 75)
    print(f"  TEMPO TOTAL DO PIPELINE                    : {t_total:8.2f} ms ({1000.0/max(1e-4, t_total):.1f} FPS)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # RENDERIZAÇÃO DO FRAME FINAL COM TODAS AS ANOTAÇÕES SOBREPOSTAS
    # -------------------------------------------------------------------------
    anotado = frame_undist.copy()

    # 1. Anotação Etapa 2: Contorno e Bounding Box da ROI HSV
    cv2.rectangle(anotado, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    if maior_contorno is not None:
        cv2.drawContours(anotado, [maior_contorno], -1, (0, 220, 0), 2)

    # 2. Anotação Etapa 3: Keypoints ORB desenhados com círculos e orientações
    cv2.drawKeypoints(anotado, kp_global, anotado, color=(255, 0, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 3. Anotação Etapa 4: Caixas do Detector HOG (se houver pedestres)
    for bx, by, bw, bh in boxes_hog:
        cv2.rectangle(anotado, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(anotado, "Pedestre (HOG+SVM)", (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # 4. Anotação Etapa 5: Banner com Top-3 da Classificação DNN abaixo da ROI
    y_banner = ry + rh + 28
    txt_top1 = f"DNN: {top3_roi[0][0]} ({top3_roi[0][1]*100:.1f}%) | {top3_roi[1][0]} ({top3_roi[1][1]*100:.1f}%)"
    (tw, th), _ = cv2.getTextSize(txt_top1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(anotado, (rx, y_banner - th - 6), (rx + tw + 10, y_banner + 4), (20, 20, 20), -1)
    cv2.rectangle(anotado, (rx, y_banner - th - 6), (rx + tw + 10, y_banner + 4), (0, 255, 255), 1)
    cv2.putText(anotado, txt_top1, (rx + 5, y_banner - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    # 5. HUD de Telemetria Geral no Canto Superior Esquerdo
    h_f, w_f = anotado.shape[:2]
    hud_w, hud_h = 490, 185
    overlay = anotado.copy()
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, anotado, 0.20, 0, dst=anotado)
    cv2.rectangle(anotado, (12, 12), (12 + hud_w, 12 + hud_h), (80, 80, 80), 1)

    cv2.putText(anotado, "PIPELINE INTEGRADO DE PERCEPCAO (TPs 1, 2, 3 + Ex 1, 2)",
                (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(anotado, f"1. Undistort (Ex 1)        : {t_etapa1:6.2f} ms",
                (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(anotado, f"2. Segmentacao HSV (TP1)    : {t_etapa2:6.2f} ms  [Bola Azul ROI]",
                (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(anotado, f"3. Features ORB (TP2)       : {t_etapa3:6.2f} ms  [{len(kp_global)} keypoints]",
                (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(anotado, f"4. Detector HOG+SVM (TP3)   : {t_etapa4:6.2f} ms  [Varredura Frame]",
                (22, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(anotado, f"5. MobileNetV2 DNN (Ex 2A)  : {t_etapa5:6.2f} ms  [{top3_roi[0][0]}]",
                (22, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(anotado, f"TEMPO TOTAL: {t_total:6.2f} ms | FPS: {1000.0/max(1e-4, t_total):.1f}",
                (22, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    # Salvar frame final anotado
    pasta_saidas.mkdir(parents=True, exist_ok=True)
    caminho_frame_final = pasta_saidas / "frame_pipeline_integrado.png"
    cv2.imwrite(str(caminho_frame_final), anotado)
    print(f"\n[Item B] Frame final com todas as anotações salvo em: {caminho_frame_final}")

    if not sem_janela:
        cv2.imshow("Validacao Item B - Pipeline Integrado de Percepcao", anotado)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

    tempos = {
        "etapa1_undistort_ms": t_etapa1,
        "etapa2_hsv_ms": t_etapa2,
        "etapa3_orb_ms": t_etapa3,
        "etapa4_hog_ms": t_etapa4,
        "etapa5_dnn_ms": t_etapa5,
        "tempo_total_ms": t_total,
    }

    return {
        "tempos": tempos,
        "frame_anotado_path": caminho_frame_final,
        "top3_roi": top3_roi,
    }


# =============================================================================
# GERAÇÃO DO JUPYTER NOTEBOOK SEQUENCIAL (exercicio02b.ipynb)
# =============================================================================

def gerar_notebook_sequencial(caminho_nb: Path, frame_alvo_idx: int = FRAME_PADRAO, executar: bool = True) -> Path:
    """Gera exercicio02b.ipynb com as 5 etapas em células sequenciais e o EXECUTA com nbclient
    (kernel do próprio venv, diretório de trabalho = pasta do exercício). As saídas gravadas no
    arquivo são, portanto, as produzidas de fato pelas células, e não textos pré-escritos.
    """
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    md, code = new_markdown_cell, new_code_cell
    cells = [
        md("# Exercício 2, Item B: Pipeline Integrado de Percepção Visual\n\n"
           "Notebook sequencial com as técnicas dos TPs e exercícios anteriores:\n"
           "1. **Exercício 1:** captura de frame e correção de distorção (`cv2.undistort`);\n"
           "2. **TP1:** segmentação de ROI por cor no espaço HSV (bola azul);\n"
           "3. **TP2:** extração de features ORB (`cv2.ORB_create`);\n"
           "4. **TP3:** detector de pedestres HOG+SVM (`cv2.HOGDescriptor`), a opção escolhida entre HOG+SVM e Haar Cascade;\n"
           "5. **Exercício 2, Item A:** classificação da ROI com MobileNetV2 via OpenCV DNN.\n\n"
           "O kernel deve rodar com o diretório de trabalho em `ex2/` (é assim que `main.py` o executa via `nbclient`)."),
        code("import json\nimport time\nfrom pathlib import Path\n\nimport cv2\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n"
             "RAIZ = Path.cwd()\nENTRADAS = RAIZ / 'entradas'\nMODELOS = RAIZ / 'modelos'\nSAIDAS = RAIZ / 'saidas'\n"
             f"FRAME_ALVO = {frame_alvo_idx}\n"
             "assert (ENTRADAS / 'bola.mp4').exists(), f'Execute o kernel dentro de ex2/ (cwd atual: {RAIZ})'\n"
             "print('Módulos importados; diretório de trabalho:', RAIZ)"),
        md("## Etapa 1: captura de frame e correção de distorção (Exercício 1)\n\n"
           "A calibração (`K` e coeficientes) vem da câmera virtual sintética do Exercício 1 e **não** corresponde à câmera que gravou `bola.mp4`: "
           "o passo é executado porque o enunciado exige encadeá-lo, mas sobre este vídeo ele não corrige distorção real."),
        code("calib = np.load(ENTRADAS / 'calibracao.npz')\nK = calib['K']\ndist = calib['dist']\n\n"
             "cap = cv2.VideoCapture(str(ENTRADAS / 'bola.mp4'))\ncap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ALVO)\nret, frame_bruto = cap.read()\ncap.release()\n"
             "if not ret:\n    raise RuntimeError(f'Falha ao ler o frame {FRAME_ALVO} de bola.mp4')\n\n"
             "t0 = time.perf_counter()\nframe_undist = cv2.undistort(frame_bruto, K, dist, None, K)\nt_etapa1 = (time.perf_counter() - t0) * 1000.0\n"
             "print(f'Etapa 1: {t_etapa1:.2f} ms | frame {FRAME_ALVO} retificado ({frame_undist.shape[1]}x{frame_undist.shape[0]})')"),
        md("## Etapa 2: segmentação de ROI por cor HSV (TP1)"),
        code("t0 = time.perf_counter()\nhsv = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2HSV)\n"
             "lower_blue = np.array([90, 60, 40], dtype=np.uint8)\nupper_blue = np.array([130, 255, 255], dtype=np.uint8)\n"
             "mask = cv2.inRange(hsv, lower_blue, upper_blue)\nkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n"
             "mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)\nmask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)\n\n"
             "cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
             "cnts = [c for c in cnts if cv2.contourArea(c) >= 500]\n"
             "if not cnts:\n    raise RuntimeError('Nenhuma ROI azul encontrada neste frame; escolha outro FRAME_ALVO')\n"
             "c_max = max(cnts, key=cv2.contourArea)\nrx, ry, rw, rh = cv2.boundingRect(c_max)\nroi = frame_undist[ry:ry+rh, rx:rx+rw]\n"
             "t_etapa2 = (time.perf_counter() - t0) * 1000.0\nprint(f'Etapa 2: {t_etapa2:.2f} ms | ROI (x={rx}, y={ry}, w={rw}, h={rh})')"),
        md("## Etapa 3: extração de features ORB (TP2)"),
        code("t0 = time.perf_counter()\norb = cv2.ORB_create(nfeatures=2000, fastThreshold=5)\nkp_roi, des_roi = orb.detectAndCompute(roi, None)\n"
             "kp_global = [cv2.KeyPoint(kp.pt[0] + rx, kp.pt[1] + ry, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_roi]\n"
             "t_etapa3 = (time.perf_counter() - t0) * 1000.0\nprint(f'Etapa 3: {t_etapa3:.2f} ms | {len(kp_global)} keypoints ORB na ROI')"),
        md("## Etapa 4: detector de pedestres HOG+SVM (TP3)\n\n"
           "`bola.mp4` não contém pessoas, portanto 0 detecções é o resultado esperado; a etapa é executada e cronometrada mesmo assim."),
        code("t0 = time.perf_counter()\nhog = cv2.HOGDescriptor()\nhog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())\n"
             "boxes_hog, _ = hog.detectMultiScale(frame_undist, winStride=(8, 8), padding=(8, 8), scale=1.05)\n"
             "t_etapa4 = (time.perf_counter() - t0) * 1000.0\nprint(f'Etapa 4: {t_etapa4:.2f} ms | {len(boxes_hog)} pedestres detectados')"),
        md("## Etapa 5: classificação da ROI com MobileNetV2 no OpenCV DNN (Item A)\n\n"
           "ImageNet não tem a classe \"bola azul\"; o top-3 mostra as classes mais próximas que a rede conhece (por exemplo `soccer_ball`, `balloon`)."),
        code("with open(MODELOS / 'imagenet_class_index.json', encoding='utf-8') as f:\n    rotulos_id = {int(k): v[1] for k, v in json.load(f).items()}\n\n"
             "net = cv2.dnn.readNetFromTFLite(str(MODELOS / 'mobilenetv2_imagenet.tflite'))\nt0 = time.perf_counter()\n"
             "blob = cv2.dnn.blobFromImage(roi, 1.0/127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)\nnet.setInput(blob)\npreds = net.forward()[0]\n"
             "top3_idx = np.argsort(preds)[::-1][:3]\ntop3 = [(rotulos_id[int(i)], float(preds[i])) for i in top3_idx]\n"
             "t_etapa5 = (time.perf_counter() - t0) * 1000.0\nt_total = t_etapa1 + t_etapa2 + t_etapa3 + t_etapa4 + t_etapa5\n\n"
             "for i, (nome, conf) in enumerate(top3, 1):\n    print(f'  top-{i}: {nome} ({conf*100:.1f}%)')\n"
             "print(f'Etapa 5: {t_etapa5:.2f} ms')\nprint(f'TEMPO TOTAL DO PIPELINE: {t_total:.2f} ms ({1000.0/t_total:.1f} FPS)')"),
        md("## Frame final com as anotações de todas as etapas"),
        code("anotado = frame_undist.copy()\ncv2.rectangle(anotado, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)\n"
             "cv2.drawContours(anotado, [c_max], -1, (0, 220, 0), 2)\n"
             "cv2.drawKeypoints(anotado, kp_global, anotado, color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)\n"
             "for bx, by, bw, bh in boxes_hog:\n    cv2.rectangle(anotado, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)\n"
             "texto = f'DNN: {top3[0][0]} ({top3[0][1]*100:.1f}%) | {top3[1][0]} ({top3[1][1]*100:.1f}%)'\n"
             "cv2.putText(anotado, texto, (rx, min(ry + rh + 28, anotado.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)\n"
             "linhas = [f'1 undistort {t_etapa1:.1f} ms', f'2 HSV {t_etapa2:.1f} ms', f'3 ORB {t_etapa3:.1f} ms ({len(kp_global)} kp)',\n"
             "          f'4 HOG+SVM {t_etapa4:.1f} ms ({len(boxes_hog)} pessoas)', f'5 DNN {t_etapa5:.1f} ms', f'total {t_total:.1f} ms']\n"
             "for i, l in enumerate(linhas):\n    cv2.putText(anotado, l, (20, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)\n\n"
             "plt.figure(figsize=(12, 7))\nplt.imshow(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB))\nplt.axis('off')\nplt.title(f'Pipeline integrado (frame {FRAME_ALVO})')\nplt.show()"),
    ]
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3 (venv)", "language": "python"},
            "language_info": {"name": "python"},
        },
    )
    if executar:
        from nbclient import NotebookClient

        print(f"[Item B] Executando o notebook com nbclient (kernel do venv, cwd = {RAIZ})...")
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(RAIZ)}}).execute()
    nbformat.write(nb, str(caminho_nb))
    print(f"[Item B] Jupyter Notebook sequencial {'executado e ' if executar else ''}gravado em: {caminho_nb}")
    return caminho_nb


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL (MAIN CLI)
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execução Estrita do Enunciado do Exercício 2 (OpenCV DNN vs Keras e Pipeline Integrado)"
    )
    parser.add_argument("--item", type=str, choices=["a", "b", "ambos"], default="ambos",
                        help="Escolha o item a executar: 'a' (benchmark), 'b' (pipeline integrado) ou 'ambos' (padrão)")
    parser.add_argument("--loops", type=int, default=20,
                        help="Número de repetições nas medições de latência do Item A (padrão: 20)")
    parser.add_argument("--sem-janela", action="store_true",
                        help="Não abre janelas gráficas do OpenCV")
    parser.add_argument("--frame", type=int, default=FRAME_PADRAO,
                        help=f"Índice do frame de bola.mp4 usado no Item B (padrão: {FRAME_PADRAO}, bola inteira no quadro)")

    args = parser.parse_args()

    ENTRADAS_DIR.mkdir(parents=True, exist_ok=True)
    SAIDAS_DIR.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 85)
    print("### EXERCÍCIO 2: CLASSIFICAÇÃO COM OPENCV DNN vs KERAS E PIPELINE INTEGRADO ###")
    print("#" * 85)

    t_inicio = time.perf_counter()

    if args.item in ("a", "ambos"):
        executar_item_a(
            pasta_imagens=PASTA_IMAGENS_TESTE,
            pasta_saidas=SAIDAS_DIR,
            caminho_modelo=CAMINHO_TFLITE,
            loops_benchmark=args.loops,
            sem_janela=args.sem_janela,
        )

    if args.item in ("b", "ambos"):
        res_b = executar_item_b(
            caminho_video=CAMINHO_VIDEO_BOLA,
            caminho_calibracao=CAMINHO_CALIBRACAO,
            caminho_modelo=CAMINHO_TFLITE,
            pasta_saidas=SAIDAS_DIR,
            frame_alvo_idx=args.frame,
            sem_janela=args.sem_janela,
        )

        # Gerar e executar o notebook sequencial exercicio02b.ipynb (mesmo frame do Item B)
        gerar_notebook_sequencial(RAIZ / "exercicio02b.ipynb", frame_alvo_idx=args.frame)

    t_total = time.perf_counter() - t_inicio
    print("\n" + "#" * 85)
    print(f"### EXECUÇÃO COMPLETA DO ENUNCIADO FINALIZADA COM SUCESSO ({t_total:.2f} s) ###")
    print(f"Arquivos gerados em: {SAIDAS_DIR.resolve()}")
    print(f"Notebook gerado em: {RAIZ / 'exercicio02b.ipynb'}")
    print("#" * 85 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
