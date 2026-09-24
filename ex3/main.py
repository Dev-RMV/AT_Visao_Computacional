#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Resolução Completa e Estrita do Enunciado do Trabalho de Visão Computacional

REQUISITOS TÉCNICOS IMPLEMENTADOS:
=================================
Item A:
  1. Detecção em tempo real com YOLOv4-tiny e SSD MobileNetV2, ambos via OpenCV DNN (cv2.dnn).
  2. Processamento de vídeo com bounding boxes, rótulos e confidências.
  3. Non-Maximum Suppression (NMS) com threshold estritamente igual a 0.4.
  4. Medição de FPS médio e latência por frame para cada modelo.
  5. Tabela comparativa no terminal: FPS, número de parâmetros e tamanho do arquivo em disco.
  6. Conclusão justificada no código e no terminal sobre qual modelo é mais adequado para robótica embarcada.
  7. Validação: gravação dos feeds de vídeo e frames com bounding boxes para YOLO e SSD.

Item B:
  1. Integração do detector YOLOv4-tiny do item A com rastreador por IoU entre frames consecutivos.
  2. Atribuição de ID persistente a cada objeto detectado.
  3. Manutenção e renderização da trilha dos últimos 30 frames como linha colorida.
  4. Detecção de entradas e saídas de objetos com impressão e overlay de contagem cumulativa.
  5. Medição e exibição no terminal da taxa de ID switches por minuto de vídeo (estimativa heurística,
     sem gabarito; limitações documentadas na classe RastreadorIoU).
  6. Discussão técnica e ética sobre o uso para contagem de pedestres em drone de vigilância urbana,
     salva em relatório Markdown ('relatorio_drone_etica.md') e impressa no terminal.
  7. Validação: feed com IDs, trilhas coloridas e HUD de contagem cumulativa.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar stdout e stderr para UTF-8 no Windows (mesmo critério dos exercícios 1, 2 e 4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Diretórios base
RAIZ = Path(__file__).resolve().parent
MODELOS_DIR = RAIZ / "modelos"
ENTRADAS_DIR = RAIZ / "entradas"
SAIDAS_DIR = RAIZ / "saidas"

# Mapeamento do COCO 91 classes (TensorFlow SSD MobileNetV2 usa IDs 1..90 com lacunas)
COCO_91_SEM_CLASSE = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}


# =============================================================================
# UTILITÁRIOS GERAIS E CORES
# =============================================================================

def carregar_coco_names(caminho: Optional[Path] = None) -> List[str]:
    """Carrega os 80 nomes de classes do dataset COCO."""
    caminho = caminho or (MODELOS_DIR / "coco.names")
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de classes não encontrado em {caminho}")
    return [l.strip() for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def mapa_coco_91(nomes80: List[str]) -> Dict[int, str]:
    """Cria dicionário id_tf -> nome_classe para modelos TensorFlow."""
    mapa: Dict[int, str] = {}
    k = 0
    for i in range(1, 91):
        if i in COCO_91_SEM_CLASSE:
            continue
        if k < len(nomes80):
            mapa[i] = nomes80[k]
            k += 1
    return mapa


def cor_por_id(track_id: int) -> Tuple[int, int, int]:
    """Gera uma cor BGR única e consistente baseada no ID do objeto (usando proporção áurea)."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.90, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def cor_por_classe(nome_classe: str) -> Tuple[int, int, int]:
    """Gera uma cor BGR determinística baseada no nome da classe."""
    h = int(hashlib.md5(nome_classe.encode("utf-8")).hexdigest()[:6], 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def contar_parametros_net(net: cv2.dnn.Net) -> int:
    """Calcula o número total de parâmetros da rede somando os pesos nos blobs das camadas."""
    total = 0
    for layer_id in range(1, len(net.getLayerNames()) + 1):
        for blob in net.getLayer(layer_id).blobs:
            total += int(np.prod(blob.shape))
    return total


def tamanho_arquivo_mb(caminho: Path) -> float:
    """Retorna o tamanho do arquivo em Megabytes (MB)."""
    return caminho.stat().st_size / (1024 * 1024)


def iou(box_a: List[int | float], box_b: List[int | float]) -> float:
    """Calcula Intersection over Union (IoU) entre duas caixas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def desenhar_deteccoes(frame: np.ndarray, dets: List[Dict[str, Any]], titulo: Optional[str] = None) -> np.ndarray:
    """Desenha bounding boxes com rótulos e confidências sobre uma cópia do frame."""
    out = frame.copy()
    w_img = out.shape[1]
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cor = cor_por_classe(d["cls"])
        cv2.rectangle(out, (x1, y1), (x2, y2), cor, 2)
        rotulo = f"{d['cls']} {d['conf'] * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yt = y1 - 4 if y1 - th - 6 >= 0 else y1 + th + 6
        xt = min(x1, max(0, w_img - tw - 6))  # mantém o rótulo dentro do quadro na borda direita
        cv2.rectangle(out, (xt, yt - th - 3), (xt + tw + 4, yt + 2), cor, -1)
        cv2.putText(out, rotulo, (xt + 2, yt - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if titulo:
        cv2.putText(out, titulo, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, titulo, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# =============================================================================
# ITEM A: DETECTORES OPENCV DNN (YOLOv4-tiny e SSD MobileNetV2)
# =============================================================================

class DetectorYOLOv4Tiny:
    """
    Detector YOLOv4-tiny (Darknet) executado via OpenCV DNN (cv2.dnn).
    Aplica NMS estritamente com threshold 0.4 e confiança mínima 0.5.
    """
    def __init__(self, cfg_path: Path, weights_path: Path, names_path: Path,
                 conf_min: float = 0.5, nms_thr: float = 0.4, tam: int = 416):
        self.nome = "YOLOv4-tiny"
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.conf_min = conf_min
        self.nms_thr = nms_thr
        self.tam = tam

        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.out_layers = self.net.getUnconnectedOutLayersNames()
        self.classes = carregar_coco_names(names_path)
        self.parametros = contar_parametros_net(self.net)
        self.tamanho_mb = tamanho_arquivo_mb(self.weights_path)

    def detectar(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Executa inferência com letterboxing, decodificação das 2 camadas de saída e NMS com threshold 0.4.
        Retorna (lista_deteccoes, dicionario_latencias_ms).
        """
        h0, w0 = frame.shape[:2]
        t0 = time.perf_counter()

        # Letterbox para preservar a proporção geométrica da imagem original
        lado = max(h0, w0)
        pad_y = (lado - h0) // 2
        pad_x = (lado - w0) // 2
        quadro = np.full((lado, lado, 3), 114, dtype=np.uint8)
        quadro[pad_y:pad_y + h0, pad_x:pad_x + w0] = frame

        blob = cv2.dnn.blobFromImage(quadro, 1.0 / 255.0, (self.tam, self.tam), swapRB=True, crop=False)
        self.net.setInput(blob)
        saidas = self.net.forward(self.out_layers)
        t1 = time.perf_counter()

        boxes: List[List[int]] = []
        confs: List[float] = []
        class_ids: List[int] = []

        for out in saidas:
            scores = out[:, 5:]
            c_ids = np.argmax(scores, axis=1)
            c_confs = scores[np.arange(len(c_ids)), c_ids]
            mascara = c_confs >= self.conf_min

            for (cx, cy, bw, bh), conf, cid in zip(out[mascara, :4], c_confs[mascara], c_ids[mascara]):
                x = int((cx - bw / 2.0) * lado)
                y = int((cy - bh / 2.0) * lado)
                w = int(bw * lado)
                h = int(bh * lado)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(int(cid))

        # Aplicação rigorosa do NMS com threshold 0.4 conforme enunciado
        idxs = cv2.dnn.NMSBoxesBatched(boxes, confs, class_ids, self.conf_min, self.nms_thr)

        deteccoes: List[Dict[str, Any]] = []
        if len(idxs) > 0:
            for idx in np.asarray(idxs).flatten():
                x, y, w, h = boxes[idx]
                # Conversão de volta para o sistema de coordenadas do frame original
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w0 - 1, x + w - pad_x)
                y2 = min(h0 - 1, y + h - pad_y)
                if x2 > x1 and y2 > y1:
                    deteccoes.append({
                        "cls": self.classes[class_ids[idx]],
                        "conf": confs[idx],
                        "bbox": [x1, y1, x2, y2]
                    })

        t2 = time.perf_counter()
        latencias = {
            "inferencia_ms": (t1 - t0) * 1000.0,
            "posprocessamento_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0
        }
        return deteccoes, latencias


class DetectorSSDMobileNetV2:
    """
    Detector SSD MobileNetV2 (TensorFlow) executado via OpenCV DNN (cv2.dnn).
    Aplica NMS estritamente com threshold 0.4 e confiança mínima 0.5.
    """
    def __init__(self, pb_path: Path, pbtxt_path: Path, names_path: Path,
                 conf_min: float = 0.5, nms_thr: float = 0.4, tam: int = 300):
        self.nome = "SSD MobileNetV2"
        self.pb_path = pb_path
        self.pbtxt_path = pbtxt_path
        self.conf_min = conf_min
        self.nms_thr = nms_thr
        self.tam = tam

        self.net = cv2.dnn.readNetFromTensorflow(str(pb_path), str(pbtxt_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.mapa_classes = mapa_coco_91(carregar_coco_names(names_path))
        self.parametros = contar_parametros_net(self.net)
        self.tamanho_mb = tamanho_arquivo_mb(self.pb_path)

    def detectar(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Executa inferência com blob 300x300, decodifica saída e aplica NMS com threshold 0.4.
        Retorna (lista_deteccoes, dicionario_latencias_ms).
        """
        h0, w0 = frame.shape[:2]
        t0 = time.perf_counter()

        blob = cv2.dnn.blobFromImage(frame, size=(self.tam, self.tam), swapRB=True, crop=False)
        self.net.setInput(blob)
        saida = self.net.forward()
        t1 = time.perf_counter()

        boxes: List[List[int]] = []
        confs: List[float] = []
        class_ids: List[int] = []

        # Formato de saída SSD: [1, 1, N, 7] com cada linha [batch, class_id, score, x1, y1, x2, y2]
        for _, cid, score, x1_norm, y1_norm, x2_norm, y2_norm in saida[0, 0]:
            cid_int = int(cid)
            if score < self.conf_min or cid_int not in self.mapa_classes:
                continue
            x1 = int(x1_norm * w0)
            y1 = int(y1_norm * h0)
            x2 = int(x2_norm * w0)
            y2 = int(y2_norm * h0)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(float(score))
            class_ids.append(cid_int)

        # Aplicação rigorosa do NMS com threshold 0.4 conforme enunciado
        idxs = cv2.dnn.NMSBoxesBatched(boxes, confs, class_ids, self.conf_min, self.nms_thr)

        deteccoes: List[Dict[str, Any]] = []
        if len(idxs) > 0:
            for idx in np.asarray(idxs).flatten():
                x, y, w, h = boxes[idx]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w0 - 1, x + w)
                y2 = min(h0 - 1, y + h)
                if x2 > x1 and y2 > y1:
                    deteccoes.append({
                        "cls": self.mapa_classes[class_ids[idx]],
                        "conf": confs[idx],
                        "bbox": [x1, y1, x2, y2]
                    })

        t2 = time.perf_counter()
        latencias = {
            "inferencia_ms": (t1 - t0) * 1000.0,
            "posprocessamento_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0
        }
        return deteccoes, latencias


# =============================================================================
# ITEM B: RASTREAMENTO POR IOU COM IDS PERSISTENTES E TRILHAS
# =============================================================================

class RastreadorIoU:
    """
    Rastreador multiobjeto por associação de IoU entre frames consecutivos (predição linear de
    posição pela velocidade média; não há filtro de Kalman):
      (1) Atribui ID persistente único a cada objeto detectado e confirmado.
      (2) Mantém trilha dos últimos 30 frames como fila FIFO de coordenadas centrais.
      (3) Detecta e contabiliza entradas e saídas de objetos (contagem cumulativa).
      (4) ESTIMA ID switches por uma heurística: um track novo que nasce sobre a posição de um
          track da mesma classe encerrado há poucos frames é contado como troca de ID.
          LIMITAÇÕES: não há gabarito (ground truth) neste vídeo, então o valor não é um ID switch
          "de verdade" no sentido MOT; a heurística só vê reidentificações após perda de detecção e
          NÃO captura a troca de IDs entre dois objetos que se cruzam sem perder a detecção. O número
          tende, portanto, a subestimar os ID switches reais.
    """
    def __init__(self, iou_thr: float = 0.30, max_perdidos: int = 15,
                 min_confirmacoes: int = 2, tam_trilha: int = 30):
        self.iou_thr = iou_thr
        self.max_perdidos = max_perdidos
        self.min_confirmacoes = min_confirmacoes
        self.tam_trilha = tam_trilha

        self.proximo_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.trilhas: Dict[int, deque] = {}
        self.entradas_cumulativas = 0
        self.saidas_cumulativas = 0
        self.id_switches_total = 0
        self.historico_eventos: List[Tuple[int, str, int, str]] = []
        self.mortos_recentes: List[Tuple[int, str, List[int]]] = []
        self._chave_interna = 0

    def _centro(self, bbox: List[int | float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) // 2), int((y1 + y2) // 2)

    def _prever_posicao(self, tr: Dict[str, Any]) -> List[float]:
        """Aplica modelo de velocidade linear para predizer a posição após frames sem detecção."""
        k = tr["missing"] + 1
        dx = tr["vel"][0] * k
        dy = tr["vel"][1] * k
        x1, y1, x2, y2 = tr["bbox"]
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def _checar_id_switch(self, frame_idx: int, tr: Dict[str, Any]) -> None:
        """
        Heurística de ID switch: um novo track que nasce próximo (IoU > 0.2 ou centro dentro da caixa)
        e com a mesma classe de um track encerrado nos últimos 25 frames é contado como provável troca
        de ID. Ver limitações no docstring da classe.
        """
        cx, cy = self._centro(tr["bbox"])
        for idx_m, (f_morto, cls_m, bbox_m) in enumerate(self.mortos_recentes):
            if cls_m != tr["cls"]:
                continue
            if iou(bbox_m, tr["bbox"]) > 0.20 or (bbox_m[0] <= cx <= bbox_m[2] and bbox_m[1] <= cy <= bbox_m[3]):
                self.id_switches_total += 1
                self.historico_eventos.append((frame_idx, "id_switch", tr["id"], tr["cls"]))
                self.mortos_recentes.pop(idx_m)
                break

    def _confirmar_track(self, tr: Dict[str, Any], frame_idx: int) -> None:
        """Atribui ID oficial definitivo ao track, contabiliza entrada e inicia a trilha."""
        tr["id"] = self.proximo_id
        self.proximo_id += 1
        self.entradas_cumulativas += 1
        self.historico_eventos.append((frame_idx, "entrada", tr["id"], tr["cls"]))
        self.trilhas[tr["id"]] = deque([self._centro(tr["bbox"])], maxlen=self.tam_trilha)
        self._checar_id_switch(frame_idx, tr)

    def update(self, deteccoes: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Atualiza o estado dos tracks com base nas novas detecções do frame atual.
        Retorna a lista de tracks ativos visíveis com seus IDs persistentes.
        """
        pares: List[Tuple[float, int, int]] = []
        for ch, tr in self.tracks.items():
            bbox_prev = self._prever_posicao(tr)
            for di, d in enumerate(deteccoes):
                if d["cls"] != tr["cls"]:
                    continue
                v_iou = iou(bbox_prev, d["bbox"])
                if v_iou >= self.iou_thr:
                    pares.append((v_iou, ch, di))

        # Associação gulosa ordenada por maior IoU
        pares.sort(reverse=True, key=lambda p: p[0])
        tracks_associados = set()
        dets_associadas = set()

        for v_iou, ch, di in pares:
            if ch in tracks_associados or di in dets_associadas:
                continue
            tr = self.tracks[ch]
            c_ant = self._centro(tr["bbox"])
            c_novo = self._centro(deteccoes[di]["bbox"])
            k = tr["missing"] + 1
            vx = (c_novo[0] - c_ant[0]) / float(k)
            vy = (c_novo[1] - c_ant[1]) / float(k)
            tr["vel"] = (0.6 * tr["vel"][0] + 0.4 * vx, 0.6 * tr["vel"][1] + 0.4 * vy)

            tr["bbox"] = deteccoes[di]["bbox"]
            tr["conf"] = deteccoes[di]["conf"]
            tr["missing"] = 0
            tr["hits"] += 1
            tracks_associados.add(ch)
            dets_associadas.add(di)

            if tr["id"] is None and tr["hits"] >= self.min_confirmacoes:
                self._confirmar_track(tr, frame_idx)
            elif tr["id"] is not None:
                self.trilhas.setdefault(tr["id"], deque(maxlen=self.tam_trilha)).append(c_novo)

        # Criação de novos candidatos para detecções não associadas
        for di, d in enumerate(deteccoes):
            if di not in dets_associadas:
                self._chave_interna += 1
                novo_tr = {
                    "id": None,
                    "bbox": d["bbox"],
                    "vel": (0.0, 0.0),
                    "cls": d["cls"],
                    "conf": d["conf"],
                    "missing": 0,
                    "hits": 1,
                    "nasceu": frame_idx
                }
                self.tracks[self._chave_interna] = novo_tr
                if self.min_confirmacoes <= 1:
                    self._confirmar_track(novo_tr, frame_idx)

        # Verificação de tracks ausentes e detecção de saídas
        chaves_remover = []
        for ch, tr in list(self.tracks.items()):
            if ch in tracks_associados or tr["nasceu"] == frame_idx:
                continue
            tr["missing"] += 1
            limite = self.max_perdidos if tr["id"] is not None else 1
            if tr["missing"] > limite:
                if tr["id"] is not None:
                    self.saidas_cumulativas += 1
                    self.historico_eventos.append((frame_idx, "saida", tr["id"], tr["cls"]))
                    self.mortos_recentes.append((frame_idx, tr["cls"], tr["bbox"]))
                    self.trilhas.pop(tr["id"], None)
                chaves_remover.append(ch)

        for ch in chaves_remover:
            self.tracks.pop(ch, None)

        self.mortos_recentes = [m for m in self.mortos_recentes if frame_idx - m[0] <= 25]

        ativos = []
        for tr in self.tracks.values():
            if tr["id"] is not None and tr["missing"] == 0:
                ativos.append({
                    "id": tr["id"],
                    "cls": tr["cls"],
                    "conf": tr["conf"],
                    "bbox": tr["bbox"]
                })
        return ativos

    @property
    def objetos_presentes(self) -> int:
        return sum(1 for tr in self.tracks.values() if tr["id"] is not None and tr["missing"] == 0)


def desenhar_rastreamento(frame: np.ndarray, tracks: List[Dict[str, Any]],
                          rastreador: RastreadorIoU, taxa_idsw_min: float) -> np.ndarray:
    """
    Renderiza no frame:
      - Trilhas dos últimos 30 frames como linhas contínuas coloridas.
      - Bounding box e rótulo 'ID <id> | <classe> <conf>' na cor do ID.
      - HUD translúcido com contagem cumulativa e taxa de ID switches/minuto.
    """
    out = frame.copy()

    # (2) Mantenha a trilha dos últimos 30 frames como linha colorida
    for tid, pontos in rastreador.trilhas.items():
        pts = list(pontos)
        cor = cor_por_id(tid)
        for k in range(1, len(pts)):
            espessura = 1 + int(3.0 * k / max(1, len(pts)))
            cv2.line(out, pts[k - 1], pts[k], cor, espessura, cv2.LINE_AA)

    # (1) Bounding boxes com ID persistente
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        cor = cor_por_id(t["id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), cor, 2)
        rotulo = f"ID {t['id']} | {t['cls']} {t['conf'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yt = y1 - 4 if y1 - th - 6 >= 0 else y1 + th + 6
        cv2.rectangle(out, (x1, yt - th - 3), (x1 + tw + 4, yt + 2), cor, -1)
        cv2.putText(out, rotulo, (x1 + 2, yt - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # (3) Painel HUD de contagem cumulativa e (4) taxa de ID switches
    hud_h, hud_w = 115, 270
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (25, 25, 30), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 90), 1)

    cv2.putText(out, "SISTEMA DE RASTREAMENTO IOU", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    linhas = [
        f"Objetos Presentes: {rastreador.objetos_presentes}",
        f"Entradas Cumulativas: {rastreador.entradas_cumulativas}",
        f"Saidas Cumulativas: {rastreador.saidas_cumulativas}",
        f"ID Switches (estim.): {taxa_idsw_min:.2f} /min"
    ]
    for i, linha in enumerate(linhas, start=1):
        cv2.putText(out, linha, (18, 28 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# =============================================================================
# GERADOR DE VÍDEO SINTÉTICO (OPÇÃO PERMITIDA PELO ENUNCIADO)
# =============================================================================

CLASSES_RECORTE = {"person", "car", "truck", "bus", "bicycle", "motorbike"}
FRAME_SNAPSHOT = 60  # frame gravado como PNG de validação (ou o último, se o vídeo for mais curto)


def extrair_recortes_reais(video_fonte: Path, detector: "DetectorYOLOv4Tiny", max_recortes: int = 6,
                           passo: int = 25, conf_min: float = 0.75) -> List[Tuple[str, np.ndarray]]:
    """Extrai recortes reais (pessoas e veículos) de um vídeo usando o próprio detector.

    Bonecos desenhados com primitivas do OpenCV não são reconhecidos por YOLO nem por SSD (redes
    treinadas em fotos reais); um vídeo sintético só serve para validar detecção e rastreamento se
    contiver objetos de aparência real, por isso os recortes são colados sobre o cenário sintético.
    """
    recortes: List[Tuple[str, np.ndarray]] = []
    cap = cv2.VideoCapture(str(video_fonte))
    idx = 0
    while cap.isOpened() and len(recortes) < max_recortes:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % passo == 0:
            dets, _ = detector.detectar(frame)
            for d in sorted(dets, key=lambda d: -d["conf"]):
                if d["cls"] not in CLASSES_RECORTE or d["conf"] < conf_min:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                if (y2 - y1) >= 70 and (x2 - x1) >= 25:
                    recortes.append((d["cls"], frame[y1:y2, x1:x2].copy()))
                if len(recortes) >= max_recortes:
                    break
        idx += 1
    cap.release()
    return recortes


def gerar_video_sintetico(caminho_video: Path, n_frames: int = 150, fps: float = 25.0,
                          recortes: Optional[List[Tuple[str, np.ndarray]]] = None) -> Path:
    """Gera um vídeo sintético de rua com objetos em movimento, cruzamentos e oclusões.

    Com `recortes` (pessoas/veículos reais extraídos de outro vídeo), os objetos são colados sobre o
    cenário e os detectores os reconhecem. Sem recortes, desenha bonecos com primitivas, que os
    detectores NÃO reconhecem: nesse caso o vídeo serve apenas como teste de fumaça do pipeline.
    """
    caminho_video.parent.mkdir(parents=True, exist_ok=True)
    w, h = 960, 540
    writer = cv2.VideoWriter(str(caminho_video), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    # Trajetórias: (x inicial, velocidade em px/frame, y da base do objeto, amplitude de oscilação)
    trajetorias = [(-60, 5.5, 330, 6), (w + 30, -4.8, 340, 5), (w + 80, -8.5, 440, 0),
                   (-120, 3.2, 300, 4), (w + 200, -6.0, 470, 0), (-200, 7.0, 360, 3)]

    for i in range(n_frames):
        frame = np.full((h, w, 3), (50, 50, 55), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (w, 160), (90, 105, 120), -1)      # prédios
        cv2.rectangle(frame, (0, 160), (w, 220), (110, 110, 105), -1)   # calçada
        cv2.rectangle(frame, (0, 440), (w, h), (110, 110, 105), -1)     # calçada
        for x in range(-120, w + 120, 140):                             # faixas da pista
            cv2.line(frame, (x + (i * 3) % 140, 330), (x + 60 + (i * 3) % 140, 330), (230, 230, 230), 4)

        if recortes:
            for k, (_cls, rec) in enumerate(recortes):
                x0, vel, y_base, amp = trajetorias[k % len(trajetorias)]
                rh, rw = rec.shape[:2]
                x = int(x0 + vel * i)
                y = int(y_base - rh + amp * np.sin(i / 5.0))
                xa, ya = max(0, x), max(0, y)
                xb, yb = min(w, x + rw), min(h, y + rh)
                if xb > xa and yb > ya:
                    frame[ya:yb, xa:xb] = rec[ya - y:yb - y, xa - x:xb - x]
        else:
            # Bonecos com primitivas (não reconhecidos pelos detectores)
            p1_x = int(-40 + i * 5.5)
            p1_y = 280 + int(6 * np.sin(i / 5.0))
            if 0 <= p1_x < w - 40:
                cv2.circle(frame, (p1_x + 18, p1_y + 16), 14, (240, 140, 20), -1)
                cv2.rectangle(frame, (p1_x + 10, p1_y + 30), (p1_x + 26, p1_y + 80), (240, 140, 20), -1)
            p2_x = int(w + 30 - i * 4.8)
            p2_y = 285 + int(5 * np.cos(i / 6.0))
            if 0 <= p2_x < w - 40:
                cv2.circle(frame, (p2_x + 18, p2_y + 16), 14, (20, 180, 240), -1)
                cv2.rectangle(frame, (p2_x + 10, p2_y + 30), (p2_x + 26, p2_y + 80), (20, 180, 240), -1)
            c1_x = int(w + 80 - i * 8.5)
            if -140 <= c1_x < w:
                cv2.rectangle(frame, (c1_x, 385), (c1_x + 130, 430), (0, 180, 240), -1)
                cv2.rectangle(frame, (c1_x + 25, 370), (c1_x + 105, 390), (0, 180, 240), -1)

        writer.write(frame)

    writer.release()
    return caminho_video


# =============================================================================
# CONCLUSÃO TÉCNICA DERIVADA DOS DADOS COLETADOS
# =============================================================================

FPS_MINIMO_TEMPO_REAL = 15.0  # limiar adotado: robô móvel a ~1 m/s reage a um obstáculo a poucos metros


def gerar_conclusao_embarcada(resultados_yolo: Dict[str, float],
                              resultados_ssd: Dict[str, float]) -> str:
    """Conclusão derivada dos números medidos; nada é afirmado a priori.

    Critério: em robótica embarcada, memória (RAM/Flash) e energia por inferência (proporcional ao
    tamanho do modelo e ao tráfego de memória) costumam ser a restrição dominante, desde que o
    detector atinja a taxa mínima de operação (FPS_MINIMO_TEMPO_REAL). Se os dois modelos passam
    do limiar, escolhe-se o menor; se só um passa, escolhe-se o que passa; se nenhum passa,
    escolhe-se o mais rápido. O trade-off contrário é sempre explicitado.
    """
    fps_min = FPS_MINIMO_TEMPO_REAL
    fy, fs = resultados_yolo["fps"], resultados_ssd["fps"]
    ly, ls = resultados_yolo["latencia_ms"], resultados_ssd["latencia_ms"]
    py, ps = resultados_yolo["parametros"] / 1e6, resultados_ssd["parametros"] / 1e6
    ty, ts = resultados_yolo["tamanho_mb"], resultados_ssd["tamanho_mb"]

    mais_rapido = "YOLOv4-tiny" if fy >= fs else "SSD MobileNetV2"
    menor = "YOLOv4-tiny" if ty <= ts else "SSD MobileNetV2"
    razao_fps = max(fy, fs) / max(1e-6, min(fy, fs))
    razao_tam = max(ty, ts) / max(1e-6, min(ty, ts))
    razao_par = max(py, ps) / max(1e-6, min(py, ps))

    if fy >= fps_min and fs >= fps_min:
        escolhido = menor
        motivo = (f"os dois modelos superam o limiar de {fps_min:.0f} FPS na CPU de teste, logo a taxa não é o gargalo; "
                  f"o critério decisivo passa a ser memória/energia, em que o {menor} é {razao_tam:.1f}x menor em disco "
                  f"e tem {razao_par:.1f}x menos parâmetros")
    elif fy >= fps_min or fs >= fps_min:
        escolhido = "YOLOv4-tiny" if fy >= fps_min else "SSD MobileNetV2"
        motivo = f"só o {escolhido} atinge o limiar de {fps_min:.0f} FPS na CPU de teste"
    else:
        escolhido = mais_rapido
        motivo = f"nenhum dos dois atinge {fps_min:.0f} FPS na CPU de teste; o {mais_rapido} é o menos lento"

    trade_off = ""
    if escolhido != mais_rapido:
        trade_off = (f"\n   Trade-off reconhecido: o {mais_rapido} foi {razao_fps:.2f}x mais rápido "
                     f"({max(fy, fs):.1f} contra {min(fy, fs):.1f} FPS). Se a latência for a restrição dominante "
                     f"(veículo rápido, ou CPU embarcada em que o {escolhido} caia abaixo de {fps_min:.0f} FPS), a escolha se inverte.")

    texto = f"""
========================================================================================
CONCLUSÃO TÉCNICA (DERIVADA DOS DADOS COLETADOS): ESCOLHA PARA ROBÓTICA EMBARCADA (ITEM A)
========================================================================================

Medições na mesma CPU, mesmo vídeo e mesmo NMS (0.4):

1. RECURSOS (FLASH / RAM DE PESOS):
   - YOLOv4-tiny     : {py:.2f} M parâmetros | {ty:.2f} MB em disco
   - SSD MobileNetV2 : {ps:.2f} M parâmetros | {ts:.2f} MB em disco
   -> O {menor} é {razao_tam:.1f}x menor em disco e tem {razao_par:.1f}x menos parâmetros.

2. TEMPO (FPS E LATÊNCIA MÉDIA POR FRAME, inferência + pós-processamento):
   - YOLOv4-tiny     : {fy:.2f} FPS ({ly:.2f} ms)
   - SSD MobileNetV2 : {fs:.2f} FPS ({ls:.2f} ms)
   -> O {mais_rapido} é {razao_fps:.2f}x mais rápido. Limiar adotado para tempo real: {fps_min:.0f} FPS
      (YOLO {'atinge' if fy >= fps_min else 'NÃO atinge'}; SSD {'atinge' if fs >= fps_min else 'NÃO atinge'}).

3. CONCLUSÃO: modelo mais adequado para robótica embarcada = **{escolhido}**
   Justificativa: {motivo}.{trade_off}
   Observação: o FPS acima exclui leitura, desenho e gravação do vídeo; em placas embarcadas (ARM, sem
   AVX2) a latência cresce e deve ser remedida no hardware alvo antes da decisão final.
========================================================================================
"""
    return texto.strip()


# =============================================================================
# RELATÓRIO TÉCNICO E ÉTICO: DRONE DE VIGILÂNCIA URBANA
# =============================================================================

def gerar_relatorio_etica_drone(taxa_idsw_min: float, entradas: int, saidas: int, id_switches: int,
                                duracao_min: float, latencia_rastreador_ms: float) -> str:
    """Gera o texto da discussão técnica e ética exigida pelo Item B (Markdown), com os números medidos."""
    relatorio = f"""# Relatório Técnico e Considerações Éticas: Contagem de Pedestres por Drone em Vigilância Urbana

**Disciplina:** Visão Computacional — Avaliação Técnica (Exercício 3, Item B)

---

## 1. Visão Geral da Arquitetura do Sistema
O sistema integra um detector de objetos em tempo real (**YOLOv4-tiny via OpenCV DNN**) com um rastreador por associação espacial (**IoU entre frames consecutivos, com predição linear de posição pela velocidade média; não há filtro de Kalman**) que atribui identificadores persistentes e preserva trilhas dos últimos 30 frames. Aplicado a aeronaves remotamente pilotadas (drones/RPAS) para monitoramento urbano, o pipeline realiza:
1. **Detecção frame a frame:** localização de pedestres e veículos com supressão de não-máximos (*NMS threshold 0.4*).
2. **Associação temporal:** IoU entre a caixa prevista de cada track e as detecções do frame atual.
3. **Contagem orientada a eventos:** entradas (track confirmado após 2 detecções) e saídas (track encerrado após 15 frames sem detecção), acumuladas.
4. **Métrica de estabilidade (estimativa heurística):** {id_switches} prováveis ID switches em {duracao_min:.2f} min de vídeo ({taxa_idsw_min:.2f}/min), com {entradas} entradas e {saidas} saídas. A heurística conta apenas um objeto reidentificado sobre a posição de um track encerrado há poucos frames; trocas de ID entre dois objetos que se cruzam sem perder a detecção não são capturadas e não há gabarito (*ground truth*) para este vídeo, logo o valor tende a subestimar os ID switches reais. Custo do rastreador: {latencia_rastreador_ms:.3f} ms por frame.

---

## 2. Aplicação Técnica em Drones de Vigilância Urbana
O uso de drones equipados com este pipeline possibilita:
- **Gestão de grandes eventos e mobilidade:** mapeamento de densidade de multidões, identificação de gargalos de circulação em tempo real e apoio a rotas de evacuação em emergências.
- **Planejamento urbano:** estimativa de fluxo de pedestres em cruzamentos e praças para dimensionamento de faixas de travessia e calçadas.
- **Desafios específicos do ponto de vista aéreo (*bird's-eye view*):**
  - *Egomotion (movimento da câmera):* drones sofrem translação e guinada com o vento. Em voo pairado (*hovering*), a associação por IoU a taxas altas (>25 FPS) se mantém; com translação, é preciso compensar o movimento de fundo (fluxo óptico ou transformação afim entre frames) antes de associar.
  - *Escala reduzida:* pedestres vistos de altitudes elevadas têm menos de 30×30 pixels, exigindo sensores de alta resolução ou redes adaptadas a objetos pequenos; o YOLOv4-tiny a 416 px perde esses alvos.
  - *Restrição energética e térmica:* o processamento deve ocorrer a bordo com limite de potência (ordem de 5 a 15 W em NPUs como Jetson Orin Nano ou Hailo-8) para não comprometer a autonomia de voo (tipicamente 20 a 35 minutos).

---

## 3. Considerações Éticas e Conformidade Regulatória (LGPD / GDPR)

A vigilância urbana por plataformas aéreas suscita preocupações sérias de privacidade e liberdades civis:

### A. Privacidade por design (*privacy by design*) e minimização de dados
- **Processamento na borda (*edge computing*):** o vídeo bruto **não deve ser transmitido nem gravado** na íntegra em servidores centrais; inferência e rastreamento ocorrem a bordo.
- **Descarte imediato de frames:** após gerar os metadados agregados (por exemplo, "12 pessoas por minuto na zona sul"), os quadros são descartados da memória volátil.
- **Ausência de reconhecimento biométrico:** o pipeline opera com caixas delimitadoras de silhuetas, **sem reconhecimento facial** nem identificação civil.

### B. Riscos de vigilância em massa e efeito inibidor (*chilling effect*)
- A presença visível de drones pode constranger o exercício legítimo de reunião pública pacífica e a liberdade de locomoção.
- O emprego deve ser vinculado a finalidades legítimas e proporcionais de segurança e mobilidade, auditadas por conselhos de cidadania e pela autoridade de proteção de dados.

### C. Auditoria de vieses e erros operacionais
- Modelos treinados em visadas horizontais apresentam mais falsos negativos em tomadas zenitais, sob baixa iluminação, oclusão por copas de árvores ou sombras longas.
- Decisões automatizadas baseadas em contagens (por exemplo, fechar acessos por superlotação aparente) exigem supervisão humana (*human-in-the-loop*), sobretudo porque a própria métrica de estabilidade acima é uma estimativa.

### D. Transparência e notificação pública
- Operações de monitoramento aéreo em áreas urbanas devem ser precedidas de sinalização visível e aviso público sobre finalidade, órgão responsável e canais de contato do encarregado de proteção de dados.

---

## 4. Conclusão
O sistema é adequado para contagem quantitativa agregada com baixo custo computacional, desde que suas limitações (ausência de reidentificação por aparência, métrica de ID switch estimada, sensibilidade a escala e egomotion) sejam declaradas. A conformidade ética depende de anonimização arquitetural: a visão computacional como instrumento de medição estatística urbana, nunca de vigilância e identificação individual.
"""
    return relatorio.strip()


# =============================================================================
# FLUXO DE EXECUÇÃO: ITEM A E ITEM B
# =============================================================================

def _processar_video_detector(detector, video_path: Path, saida_video: Path, saida_frame: Path,
                              titulo: str, max_frames: int, sem_janela: bool) -> Tuple[List[float], int, int]:
    """Roda um detector sobre o vídeo, grava o feed anotado e retorna (latências, total de detecções, frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo em {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(saida_video), cv2.VideoWriter_fourcc(*"XVID"), fps_in, (w, h))

    latencias: List[float] = []
    total_dets = 0
    frame_idx = 0
    ultimo_vis: Optional[np.ndarray] = None
    snapshot_salvo = False

    while max_frames <= 0 or frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        dets, tempos = detector.detectar(frame)
        latencias.append(tempos["total_ms"])
        total_dets += len(dets)

        vis = desenhar_deteccoes(frame, dets, f"{titulo} | Frame {frame_idx} | {tempos['total_ms']:.1f}ms")
        writer.write(vis)
        ultimo_vis = vis
        if frame_idx == FRAME_SNAPSHOT:
            cv2.imwrite(str(saida_frame), vis)
            snapshot_salvo = True
        if not sem_janela:
            cv2.imshow(titulo, vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                print("         Interrompido pelo usuário.")
                break
        frame_idx += 1

    cap.release()
    writer.release()
    if not sem_janela:
        cv2.destroyAllWindows()
    if not snapshot_salvo and ultimo_vis is not None:
        cv2.imwrite(str(saida_frame), ultimo_vis)
    return latencias, total_dets, frame_idx


def executar_item_a(video_path: Path, max_frames: int = 0, sem_janela: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Executa o Item A:
      - Processa o vídeo com YOLOv4-tiny e SSD MobileNetV2 (OpenCV DNN), com NMS 0.4.
      - Mede FPS médio e latência por frame (inferência + pós-processamento).
      - Grava os feeds com bounding boxes e um frame PNG de cada modelo.
      - Imprime a tabela comparativa e a conclusão derivada dos dados.
    """
    print("\n" + "=" * 80)
    print("INICIANDO EXECUÇÃO DO ITEM A: DETECÇÃO COM YOLOV4-TINY E SSD MOBILENETV2 (OPENCV DNN)")
    print("=" * 80)

    cfg_yolo = MODELOS_DIR / "yolov4-tiny.cfg"
    weights_yolo = MODELOS_DIR / "yolov4-tiny.weights"
    names_coco = MODELOS_DIR / "coco.names"
    pb_ssd = MODELOS_DIR / "ssd_mobilenet_v2_coco.pb"
    pbtxt_ssd = MODELOS_DIR / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

    print("[Item A] Carregando YOLOv4-tiny via OpenCV DNN...")
    detector_yolo = DetectorYOLOv4Tiny(cfg_yolo, weights_yolo, names_coco, conf_min=0.5, nms_thr=0.4)
    print(f"         Parâmetros: {detector_yolo.parametros / 1e6:.2f} M | Tamanho: {detector_yolo.tamanho_mb:.2f} MB | Entrada: {detector_yolo.tam}x{detector_yolo.tam}")

    print("[Item A] Carregando SSD MobileNetV2 via OpenCV DNN...")
    detector_ssd = DetectorSSDMobileNetV2(pb_ssd, pbtxt_ssd, names_coco, conf_min=0.5, nms_thr=0.4)
    print(f"         Parâmetros: {detector_ssd.parametros / 1e6:.2f} M | Tamanho: {detector_ssd.tamanho_mb:.2f} MB | Entrada: {detector_ssd.tam}x{detector_ssd.tam}")

    limite_txt = "todos os frames" if max_frames <= 0 else f"{max_frames} frames"
    resultados = {}
    for nome, detector, arquivo in (("YOLOv4-tiny", detector_yolo, "yolo"), ("SSD MobileNetV2", detector_ssd, "ssd")):
        print(f"\n[Item A] Processando {limite_txt} com {nome} (NMS 0.4)...")
        latencias, total_dets, n_frames = _processar_video_detector(
            detector, video_path, SAIDAS_DIR / f"item_a_{arquivo}.avi", SAIDAS_DIR / f"item_a_{arquivo}_frame.png",
            f"{nome} (DNN)", max_frames, sem_janela)
        lat_media = float(np.mean(latencias)) if latencias else 0.0
        fps = 1000.0 / lat_media if lat_media > 0 else 0.0
        print(f"         Concluído: {n_frames} frames | Latência média: {lat_media:.2f} ms | FPS médio: {fps:.2f} | Detecções: {total_dets}")
        if total_dets == 0:
            print(f"         AVISO: o {nome} não detectou nenhum objeto neste vídeo; o feed gravado não valida a detecção.")
        resultados[arquivo] = {
            "fps": fps,
            "latencia_ms": lat_media,
            "parametros": detector.parametros,
            "tamanho_mb": detector.tamanho_mb,
            "deteccoes": total_dets,
            "frames": n_frames,
        }

    res_yolo, res_ssd = resultados["yolo"], resultados["ssd"]

    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA DE DESEMPENHO (ITEM A): YOLOv4-tiny vs SSD MobileNetV2")
    print("=" * 80)
    header = f"{'Modelo':<18} | {'FPS Médio':<12} | {'Latência (ms)':<14} | {'Parâmetros (M)':<15} | {'Tam. Disco (MB)':<16} | {'Detecções':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'YOLOv4-tiny':<18} | {res_yolo['fps']:<12.2f} | {res_yolo['latencia_ms']:<14.2f} | {res_yolo['parametros'] / 1e6:<15.2f} | {res_yolo['tamanho_mb']:<16.2f} | {res_yolo['deteccoes']:<10d}")
    print(f"{'SSD MobileNetV2':<18} | {res_ssd['fps']:<12.2f} | {res_ssd['latencia_ms']:<14.2f} | {res_ssd['parametros'] / 1e6:<15.2f} | {res_ssd['tamanho_mb']:<16.2f} | {res_ssd['deteccoes']:<10d}")
    print("-" * len(header))
    print(f"(FPS = 1000 / latência média de inferência + pós-processamento; {res_yolo['frames']} frames de {video_path.name})")

    print(gerar_conclusao_embarcada(res_yolo, res_ssd))
    return res_yolo, res_ssd


def executar_item_b(video_path: Path, max_frames: int = 0, sem_janela: bool = True) -> Dict[str, Any]:
    """
    Executa o Item B:
      - Integra YOLOv4-tiny com RastreadorIoU (ID persistente, trilha de 30 frames, entradas/saídas).
      - Mede a taxa estimada de ID switches por minuto e o custo do rastreador por frame.
      - Grava o feed com IDs, trilhas e HUD de contagem, e o relatório técnico/ético em Markdown.
    """
    print("\n" + "=" * 80)
    print("INICIANDO EXECUÇÃO DO ITEM B: INTEGRAÇÃO YOLOv4-TINY COM RASTREAMENTO IOU")
    print("=" * 80)

    detector_yolo = DetectorYOLOv4Tiny(MODELOS_DIR / "yolov4-tiny.cfg", MODELOS_DIR / "yolov4-tiny.weights",
                                       MODELOS_DIR / "coco.names", conf_min=0.5, nms_thr=0.4)
    rastreador = RastreadorIoU(iou_thr=0.30, max_perdidos=15, min_confirmacoes=2, tam_trilha=30)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo em {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    saida_rastreamento_path = SAIDAS_DIR / "item_b_rastreamento.avi"
    writer_rastreamento = cv2.VideoWriter(str(saida_rastreamento_path), cv2.VideoWriter_fourcc(*"XVID"), fps_in, (w, h))

    limite_txt = "todos os frames" if max_frames <= 0 else f"{max_frames} frames"
    print(f"[Item B] Processando {limite_txt} com YOLOv4-tiny + Rastreamento IoU...")
    frame_idx = 0
    latencias_rastreador: List[float] = []
    taxa_idsw_min = 0.0

    while max_frames <= 0 or frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        dets, _ = detector_yolo.detectar(frame)
        t0 = time.perf_counter()
        tracks_ativos = rastreador.update(dets, frame_idx)
        latencias_rastreador.append((time.perf_counter() - t0) * 1000.0)

        duracao_video_min = max(1e-4, (frame_idx + 1) / (fps_in * 60.0))
        taxa_idsw_min = rastreador.id_switches_total / duracao_video_min

        vis = desenhar_rastreamento(frame, tracks_ativos, rastreador, taxa_idsw_min)
        writer_rastreamento.write(vis)

        if frame_idx in (FRAME_SNAPSHOT, 2 * FRAME_SNAPSHOT):
            cv2.imwrite(str(SAIDAS_DIR / f"item_b_rastreamento_frame_{frame_idx:04d}.png"), vis)

        if (frame_idx + 1) % 50 == 0:
            print(f"         Frame {frame_idx + 1} | Presentes: {rastreador.objetos_presentes} | "
                  f"Entradas: {rastreador.entradas_cumulativas} | Saídas: {rastreador.saidas_cumulativas} | "
                  f"ID switches (estim.): {rastreador.id_switches_total} ({taxa_idsw_min:.2f}/min)")

        if not sem_janela:
            cv2.imshow("Item B - Rastreamento IoU (YOLOv4-tiny)", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                print("         Interrompido pelo usuário.")
                break

        frame_idx += 1

    cap.release()
    writer_rastreamento.release()
    if not sem_janela:
        cv2.destroyAllWindows()

    duracao_total_min = frame_idx / (fps_in * 60.0)
    taxa_final_idsw_min = rastreador.id_switches_total / max(1e-4, duracao_total_min)
    lat_rastreador = float(np.mean(latencias_rastreador)) if latencias_rastreador else 0.0

    print("\n" + "=" * 80)
    print("MÉTRICAS FINAIS DE RASTREAMENTO (ITEM B)")
    print("=" * 80)
    print(f"Frames Processados:                 {frame_idx}")
    print(f"Duração do Trecho de Vídeo:         {duracao_total_min * 60.0:.2f} segundos ({duracao_total_min:.3f} minutos)")
    print(f"Objetos Presentes no Final:         {rastreador.objetos_presentes}")
    print(f"Total de Entradas Cumulativas:      {rastreador.entradas_cumulativas}")
    print(f"Total de Saídas Cumulativas:        {rastreador.saidas_cumulativas}")
    print(f"ID Switches (estimativa heurística): {rastreador.id_switches_total}")
    print(f"TAXA ESTIMADA DE ID SWITCHES/MIN:   {taxa_final_idsw_min:.2f} por minuto de vídeo")
    print(f"Custo do Rastreador por Frame:      {lat_rastreador:.3f} ms")
    print("Nota: a estimativa conta reidentificações sobre tracks recém-encerrados; não há gabarito neste vídeo e")
    print("      trocas de ID entre objetos que se cruzam sem perder a detecção não são capturadas (tende a subestimar).")
    if rastreador.entradas_cumulativas == 0:
        print("AVISO: nenhum objeto foi rastreado; o vídeo não contém objetos que o YOLOv4-tiny reconheça.")
    print("=" * 80)

    relatorio_md = gerar_relatorio_etica_drone(taxa_final_idsw_min, rastreador.entradas_cumulativas,
                                              rastreador.saidas_cumulativas, rastreador.id_switches_total,
                                              duracao_total_min, lat_rastreador)
    caminho_relatorio = RAIZ / "relatorio_drone_etica.md"
    caminho_relatorio.write_text(relatorio_md, encoding="utf-8")
    print(f"\n[Item B] Relatório técnico e ético gravado em: {caminho_relatorio}")

    print("\n" + "=" * 80)
    print("SÍNTESE DA DISCUSSÃO ÉTICA E USO EM DRONE DE VIGILÂNCIA URBANA:")
    print("=" * 80)
    print("""
1. FINALIDADE E MINIMIZAÇÃO:
   O sistema destina-se exclusivamente à estimativa estatística de fluxo de pedestres
   e densidade urbana (gestão de tráfego, evacuação em emergências). Não executa
   identificação biométrica nem reconhecimento facial.

2. PRIVACIDADE POR DESIGN (EDGE COMPUTING):
   Todo o processamento ocorre a bordo do drone. Nenhum frame de vídeo bruto deve
   ser transmitido para servidores remotos; apenas estatísticas agregadas (contagens)
   são reportadas, e os frames são descartados imediatamente da memória RAM.

3. CONFORMIDADE COM LGPD E GDPR:
   Anonimização estrita, Relatório de Impacto à Proteção de Dados (RIPD), notificação
   visível da população sobre áreas sobrevoadas e auditoria contínua de vieses.
""")
    print("=" * 80)

    return {
        "frames": frame_idx,
        "entradas": rastreador.entradas_cumulativas,
        "saidas": rastreador.saidas_cumulativas,
        "id_switches": rastreador.id_switches_total,
        "taxa_idsw_min": taxa_final_idsw_min,
        "duracao_min": duracao_total_min,
        "latencia_rastreador_ms": lat_rastreador,
    }


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL (MAIN)
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execução do Enunciado do Trabalho de Visão Computacional (Itens A e B)"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Caminho do vídeo de entrada (padrão: entradas/vtest.avi se existir, senão gera sintético)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Número máximo de frames a processar em cada item (padrão: 0 = todos os frames do vídeo)")
    parser.add_argument("--sintetico", action="store_true",
                        help="Gera e usa o vídeo sintético (com recortes reais de vtest.avi, se existir) mesmo que vtest.avi exista")
    parser.add_argument("--sem-janela", action="store_true",
                        help="Não abre janelas do OpenCV; apenas grava os vídeos e imagens em saidas/")

    args = parser.parse_args()

    SAIDAS_DIR.mkdir(parents=True, exist_ok=True)
    ENTRADAS_DIR.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    vtest_caminho = ENTRADAS_DIR / "vtest.avi"
    video_escolhido: Path
    if args.video and not args.sintetico:
        video_escolhido = Path(args.video)
    elif not args.sintetico and vtest_caminho.exists():
        video_escolhido = vtest_caminho
    else:
        video_escolhido = ENTRADAS_DIR / "video_sintetico.avi"
        recortes: List[Tuple[str, np.ndarray]] = []
        if vtest_caminho.exists():
            detector_tmp = DetectorYOLOv4Tiny(MODELOS_DIR / "yolov4-tiny.cfg", MODELOS_DIR / "yolov4-tiny.weights",
                                              MODELOS_DIR / "coco.names", conf_min=0.5, nms_thr=0.4)
            recortes = extrair_recortes_reais(vtest_caminho, detector_tmp)
            print(f"[Ambiente] {len(recortes)} recortes reais extraídos de vtest.avi para compor o vídeo sintético.")
        else:
            print("[Ambiente] AVISO: 'vtest.avi' não encontrado. O vídeo sintético terá bonecos desenhados, que os "
                  "detectores NÃO reconhecem; ele serve apenas como teste de fumaça do pipeline.")
        n_frames = args.max_frames if args.max_frames > 0 else 150
        print(f"[Ambiente] Gerando vídeo sintético em: {video_escolhido}")
        gerar_video_sintetico(video_escolhido, n_frames=n_frames, recortes=recortes)

    print(f"\n[Ambiente] Vídeo selecionado para validação: {video_escolhido.name}")
    print(f"[Ambiente] Limite de frames por teste: {'todos' if args.max_frames <= 0 else args.max_frames}")

    t_inicio = time.perf_counter()
    executar_item_a(video_escolhido, max_frames=args.max_frames, sem_janela=args.sem_janela)
    executar_item_b(video_escolhido, max_frames=args.max_frames, sem_janela=args.sem_janela)

    t_total = time.perf_counter() - t_inicio
    print(f"\n>>> EXECUÇÃO COMPLETA DO ENUNCIADO FINALIZADA EM {t_total:.2f} SEGUNDOS <<<")
    print(f"Vídeos e imagens de validação disponíveis na pasta: {SAIDAS_DIR}")
    print(f"Relatório ético e técnico disponível em: {RAIZ / 'relatorio_drone_etica.md'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
