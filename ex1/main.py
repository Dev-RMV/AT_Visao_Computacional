#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Resolução Completa e Estrita do Enunciado do Trabalho de Visão Computacional (Exercício 1)
===================================================================================================

ENUNCIADO IMPLEMENTADO:
-----------------------
Item A:
  1. Uso do vídeo na pasta entradas e criação de uma câmera virtual para análise contínua.
  2. Uso de ao menos 15 imagens de padrão de xadrez (9x6 cantos internos, 20 imagens sintéticas na pasta entradas).
  3. Calibração com cv2.findChessboardCorners e cv2.calibrateCamera, obtendo matriz intrínseca K,
     coeficientes de distorção e erros de reprojeção por imagem e médio.
  4. Aplicação de cv2.undistort em imagem distorcida com exibição e gravação de painel lado a lado (Original vs Corrigida).
  5. Impressão no terminal de K, dos 5 coeficientes de distorção e do erro médio de reprojeção em pixels.
  6. Comentários técnicos detalhados no código explicando o significado físico de cada parâmetro e os
     valores de erro considerados aceitáveis para aplicações robóticas.
  7. Validação: painel original x corrigido, terminal com métricas completas e comentário técnico.

Item B:
  1. Utilização da câmera virtual calibrada no Item A.
  2. Implementação de sobreposição de Realidade Aumentada (AR) em tempo real sobre vídeo gravado do tabuleiro.
  3. Detecção contínua do tabuleiro com cv2.solvePnP para estimativa da pose (vetores de rotação rvec e translação tvec).
  4. Projeção de um cubo 3D virtual com aresta de 1 quadrado sobre o canto de origem (0, 0, 0) usando cv2.projectPoints.
  5. Renderização das arestas do cubo com cores distintas por face e preenchimento semitransparente ordenado por profundidade.
  6. Exibição do vetor de rotação (rvec) e translação (tvec) no terminal a cada frame.
  7. Garantia de estabilidade do cubo ao longo do movimento da câmera (refinamento subpixel, continuidade temporal e IPPE/Iterative).
  8. Validação: feed com cubo 3D sobreposto, terminal com rvec/tvec a cada frame e estabilidade comprovada.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar stdout e stderr para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# COMENTÁRIOS TÉCNICOS: SIGNIFICADO FÍSICO DOS PARÂMETROS E CRITÉRIOS ROBÓTICOS
# =============================================================================
#
# 1. MATRIZ INTRÍNSECA DA CÂMERA (K):
#    A matriz K (3x3) modela a geometria projetiva interna da câmera (pinhole model),
#    transformando coordenadas tridimensionais no referencial da câmera (em metros)
#    para coordenadas bidimensionais no plano discreto do sensor digital (em pixels):
#
#            [[ fx,   s,  cx ],
#        K = [  0,  fy,  cy ],
#            [  0,   0,   1 ]]
#
#    - fx, fy: Distâncias focais expressas em PIXELS ao longo dos eixos X e Y.
#      Fisicamente, fx = f * mx e fy = f * my, onde:
#        * f é a distância focal óptica da lente (em milímetros ou metros);
#        * mx, my são as densidades de pixels do sensor (pixels/mm, inversos do pixel pitch sx, sy).
#      Quando o pixel é perfeitamente quadrado (sx = sy), fx = fy. Diferenças entre fx e fy
#      ocorrem quando o sensor possui pixels retangulares (anisotrópicos) ou a imagem sofreu
#      redimensionamento com aspect ratio alterado.
#      Significado físico em Robótica:
#        fx e fy determinam a escala métrica da projeção e o Campo de Visão (FOV - Field of View):
#          FOV_horizontal = 2 * arctan(largura_pixels / (2 * fx))
#          FOV_vertical   = 2 * arctan(altura_pixels  / (2 * fy))
#        Em visão estéreo e odometria visual robótica, fx e fy são cruciais para triangular profundidade Z:
#          Z = (fx * baseline) / disparidade. Um erro de 1% em fx propaga-se diretamente como 1% de erro
#          na distância calculada até um obstáculo ou objeto manipulado.
#
#    - cx, cy: Coordenadas do PONTO PRINCIPAL (em pixels).
#      Representam a interseção exata do eixo óptico principal da lente com o plano do sensor de imagem.
#      Teoricamente ficaria no centro geométrico do sensor (largura/2, altura/2). Porém, em câmeras
#      comerciais e robóticas reais, tolerâncias mecânicas na montagem do conjunto de lentes sobre a PCB
#      provocam deslocamentos de vários pixels.
#      Significado físico em Robótica:
#        Erros em cx e cy geram erros sistemáticos de mira angular. Ao calcular o vetor diretor para um
#        alvo a ser alcançado por um efetuador robótico (garras de robôs pick-and-place), o erro angular
#        decorrente de cx/cy acarreta colisão ou falha no agarre.
#
#    - s (skew / cisalhamento): Parâmetro K[0, 1] que quantifica a não ortogonalidade entre as linhas
#      e colunas de fotossítios do sensor. Em sensores digitais CMOS/CCD modernos fabricados por fotolitografia,
#      os fotossítios são rigorosamente ortogonais, portanto s = 0.
#
# 2. COEFICIENTES DE DISTORÇÃO (dist = [k1, k2, p1, p2, k3]):
#    Modelam desvios ópticos geométricos reais que violam a projeção linear pinhole (Brown-Conrady):
#
#    - k1, k2, k3: Coeficientes de DISTORÇÃO RADIAL.
#      Decorrem da curvatura esférica e refração não uniforme da lente nas bordas, sendo simétricos
#      em relação ao centro óptico e dependentes do raio normalizado r = sqrt(x^2 + y^2):
#        x_distorcido = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)
#        y_distorcido = y * (1 + k1*r^2 + k2*r^4 + k3*r^6)
#      * k1 < 0: Distorção do tipo BARRIL (Barrel), típica de lentes grande-angulares e webcams,
#        onde linhas retas periféricas curvam-se para fora (efeito olho de peixe moderado).
#      * k1 > 0: Distorção do tipo ALMOFADA (Pincushion), típica de teleobjetivas e lentes de longo alcance.
#      * k2 e k3 refinam a curvatura em distâncias maiores do centro óptico.
#      Significado físico em Robótica:
#        Se a distorção radial não for corrigida, retas do mundo real (portas, corredores, mesas)
#        são observadas como curvas, corrompendo algoritmos de extração de retas (LSD, Hough), mapeamento
#        de ocupação (occupancy grid) e recalibração visual de robôs em corredores.
#
#    - p1, p2: Coeficientes de DISTORÇÃO TANGENCIAL (Descentramento).
#      Causados pela falta de paralelismo exato entre o plano das lentes e o plano do sensor de silício:
#        x_tangencial = 2 * p1 * x * y + p2 * (r^2 + 2 * x^2)
#        y_tangencial = p1 * (r^2 + 2 * y^2) + 2 * p2 * x * y
#      Em lentes industriais de qualidade robótica, p1 e p2 são muito pequenos (~1e-4 a 1e-3).
#
# 3. CRITÉRIOS DE ERRO ACEITÁVEL PARA APLICAÇÕES ROBÓTICAS:
#    O Erro de Reprojeção Médio (Mean Reprojection Error - MRE / RMS) quantifica a discrepância média
#    (em pixels) entre os cantos detectados no sensor e as projeções matemáticas dos pontos 3D conhecidos:
#
#    - MRE < 0.5 pixels: EXCELENTE / PADRÃO OURO PARA ROBÓTICA.
#      Precisão subpixel obrigatória para:
#        * Manipulação robótica de alta precisão (pick-and-place com tolerância submilimétrica);
#        * Calibração mão-olho (hand-eye calibration eye-in-hand ou eye-to-hand);
#        * SLAM visual estéreo e monocular (ORB-SLAM, RTAB-Map);
#        * Odometria visual de drones e robôs móveis;
#        * Realidade aumentada estável, sem vibração perceptual (jitter) dos objetos virtuais.
#
#    - 0.5 <= MRE <= 1.0 pixel: BOM / ACEITÁVEL.
#      Suficiente para navegação de robôs móveis terrestres em ambientes amplos, evasão de obstáculos
#      volumosos e tarefas de inspeção visual genérica onde pequenos desvios de pose não comprometem a missão.
#
#    - MRE > 1.0 pixel: INACEITÁVEL / EXIGE RECALIBRAÇÃO.
#      Indica que a calibração sofreu com ruído excessivo de detecção de cantos, movimento da câmera,
#      iluminação inadequada, número insuficiente de ângulos ou modelo mal condicionado.
#      Consequências em robótica:
#        * Acúmulo descontrolado de drift em odometria visual;
#        * Falha na estimativa de profundidade e colisão de braços robóticos com bancadas;
#        * Trepidação severa e descolamento de objetos virtuais em Realidade Aumentada.
# =============================================================================

# Diretórios padrão
RAIZ = Path(__file__).resolve().parent
ENTRADAS_DIR = RAIZ / "entradas"
SAIDAS_DIR = RAIZ / "saidas"

# Padrão geométrico do tabuleiro sintético fornecido
PADRAO_TABULEIRO = (9, 6)   # 9 colunas internas x 6 linhas internas
LADO_QUADRADO_M = 0.025     # 25 milímetros por quadrado (0.025 metros)

# Configuração geométrica do cubo 3D virtual
# Base em Z=0 (apoiado no tabuleiro) e topo em Z=-s (projetando-se em direção à câmera)
VERTICES_CUBO_UNIT = np.float32([
    [0.0, 0.0,  0.0],  # V0: base, origem (canto 0)
    [1.0, 0.0,  0.0],  # V1: base, eixo X
    [1.0, 1.0,  0.0],  # V2: base, diagonal
    [0.0, 1.0,  0.0],  # V3: base, eixo Y
    [0.0, 0.0, -1.0],  # V4: topo, acima da origem
    [1.0, 0.0, -1.0],  # V5: topo, acima de V1
    [1.0, 1.0, -1.0],  # V6: topo, acima de V2
    [0.0, 1.0, -1.0],  # V7: topo, acima de V3
])

# 6 Faces do cubo com cores BGR distintas por face (atendendo estritamente ao Item B3)
FACES_CUBO = [
    ([0, 1, 2, 3], "Base",     (0, 200, 0)),     # Verde Esmeralda
    ([4, 5, 6, 7], "Topo",     (0, 140, 255)),   # Laranja Solar
    ([0, 1, 5, 4], "Frente",   (255, 230, 0)),   # Ciano Elétrico
    ([1, 2, 6, 5], "Direita",  (30, 30, 240)),   # Vermelho Carmesim
    ([2, 3, 7, 6], "Fundo",    (0, 215, 255)),   # Amarelo Ouro
    ([3, 0, 4, 7], "Esquerda", (240, 32, 160)),  # Magenta / Rosa Choque
]

# Eixos 3D para renderização no referencial de origem
EIXOS_3D_UNIT = np.float32([
    [0.0, 0.0,  0.0],  # Origem
    [1.5, 0.0,  0.0],  # Eixo X (+1.5 quadrados)
    [0.0, 1.5,  0.0],  # Eixo Y (+1.5 quadrados)
    [0.0, 0.0, -1.5],  # Eixo Z (-1.5 quadrados, saindo do tabuleiro)
])
CORES_EIXOS = [
    (0, 0, 255),  # X: Vermelho
    (0, 255, 0),  # Y: Verde
    (255, 0, 0),  # Z: Azul
]


# =============================================================================
# CLASSE: CÂMERA VIRTUAL CALIBRADA (ITENS A e B)
# =============================================================================

class CameraVirtual:
    """Representa a câmera virtual calibrada com suporte a processamento de vídeo,

    análise métrica, correção de distorção e Realidade Aumentada.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        pattern: Tuple[int, int] = PADRAO_TABULEIRO,
        square_m: float = LADO_QUADRADO_M,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(dist, dtype=np.float64).reshape(-1, 1)
        self.pattern = pattern
        self.square_m = float(square_m)
        self.image_size = image_size  # (largura, altura)

        # Pontos 3D do tabuleiro no referencial do mundo (Z=0, origem no primeiro canto interno)
        cols, rows = self.pattern
        obj = np.zeros((cols * rows, 3), np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self.obj_points = obj * self.square_m

        # Histórico temporal para estabilidade de pose e cubo
        self.prev_rvec: Optional[np.ndarray] = None
        self.prev_tvec: Optional[np.ndarray] = None
        self.prev_corners: Optional[np.ndarray] = None
        self.prev_proj_vertices: Optional[np.ndarray] = None
        self.prev_proj_vertices2: Optional[np.ndarray] = None

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Aplica cv2.undistort para retificar distorções ópticas do frame."""
        return cv2.undistort(frame, self.K, self.dist, None, self.K)

    def detect_chessboard(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detecta cantos internos do tabuleiro com refinamento subpixel

        e verificação de consistência de orientação temporal.
        """
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ok, corners = cv2.findChessboardCorners(gray, self.pattern, flags=flags)
        if not ok:
            return False, None

        # Refinamento subpixel (precisão crítica para estabilidade de solvePnP)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Garantir orientação consistente com frame anterior (evita inversão 180° que fliparia o cubo)
        if self.prev_corners is not None:
            dist_direta = np.linalg.norm(corners_sub[0, 0] - self.prev_corners[0, 0])
            dist_inversa = np.linalg.norm(corners_sub[-1, 0] - self.prev_corners[0, 0])
            if dist_inversa < dist_direta:
                corners_sub = corners_sub[::-1].copy()

        self.prev_corners = corners_sub.copy()
        return True, corners_sub

    def estimate_pose(self, corners: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """Estima a pose 6DoF (rvec, tvec) usando cv2.solvePnP com continuidade temporal.

        Usa cv2.SOLVEPNP_ITERATIVE com extrinsic guess quando há pose anterior válida,
        ou cv2.SOLVEPNP_IPPE (alvos planares) na inicialização ou se o erro de reprojeção subir.
        """
        rvec: Optional[np.ndarray] = None
        tvec: Optional[np.ndarray] = None
        rms_reproj = 0.0

        if self.prev_rvec is not None and self.prev_tvec is not None:
            ok, rvec_est, tvec_est = cv2.solvePnP(
                self.obj_points,
                corners,
                self.K,
                self.dist,
                self.prev_rvec.copy(),
                self.prev_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                proj, _ = cv2.projectPoints(self.obj_points, rvec_est, tvec_est, self.K, self.dist)
                err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
                if err <= 2.0:
                    rvec, tvec, rms_reproj = rvec_est, tvec_est, float(err)

        if rvec is None or tvec is None:
            ok, rvec_est, tvec_est = cv2.solvePnP(
                self.obj_points,
                corners,
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok:
                return False, None, None, 999.0
            proj, _ = cv2.projectPoints(self.obj_points, rvec_est, tvec_est, self.K, self.dist)
            rms_reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
            rvec, tvec = rvec_est, tvec_est

        self.prev_rvec = rvec.copy()
        self.prev_tvec = tvec.copy()
        return True, rvec, tvec, rms_reproj

    def project_cube(self, rvec: np.ndarray, tvec: np.ndarray, aresta_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """Projeta os vértices do cubo 3D e eixos de coordenadas para o plano da imagem."""
        pts_cubo_3d = VERTICES_CUBO_UNIT * aresta_m
        pts_2d, _ = cv2.projectPoints(pts_cubo_3d, rvec, tvec, self.K, self.dist)

        eixos_3d = EIXOS_3D_UNIT * aresta_m
        eixos_2d, _ = cv2.projectPoints(eixos_3d, rvec, tvec, self.K, self.dist)
        return pts_2d.reshape(-1, 2), eixos_2d.reshape(-1, 2)

    def render_ar_cube(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        aresta_m: float,
        frame_idx: int,
        fps_atual: float,
        rms_reproj: float,
    ) -> Tuple[np.ndarray, float]:
        """Renderiza o cubo 3D sobre o frame com cores distintas por face,

        preenchimento semitransparente em profundidade, eixos XYZ e HUD analítico.
        Retorna o frame aumentado e o valor de jitter/estabilidade dos vértices.
        """
        out = frame.copy()
        pts_2d, eixos_2d = self.project_cube(rvec, tvec, aresta_m)
        p = np.round(pts_2d).astype(np.int32)
        e = np.round(eixos_2d).astype(np.int32)

        # Cálculo da métrica de estabilidade (segunda diferença dos vértices: aceleração espúria)
        jitter = 0.0
        if self.prev_proj_vertices is not None and self.prev_proj_vertices2 is not None:
            # d2P = P(t) - 2*P(t-1) + P(t-2) -> deve ser próximo de 0 em movimento suave
            d2 = pts_2d - 2 * self.prev_proj_vertices + self.prev_proj_vertices2
            jitter = float(np.mean(np.linalg.norm(d2, axis=1)))
        self.prev_proj_vertices2 = self.prev_proj_vertices.copy() if self.prev_proj_vertices is not None else None
        self.prev_proj_vertices = pts_2d.copy()

        # Ordenar faces por profundidade (Z no referencial da câmera) - Painter's Algorithm
        R, _ = cv2.Rodrigues(rvec)
        pts_cam = (R @ (VERTICES_CUBO_UNIT * aresta_m).T + tvec.reshape(3, 1)).T
        faces_ordenadas = sorted(FACES_CUBO, key=lambda f: -float(np.mean(pts_cam[f[0], 2])))

        # 1. Preenchimento semitransparente das faces
        overlay = out.copy()
        for idxs, _, cor in faces_ordenadas:
            poly = p[idxs]
            cv2.fillConvexPoly(overlay, poly, cor)
        cv2.addWeighted(overlay, 0.40, out, 0.60, 0, dst=out)

        # 2. Desenho das arestas com cores distintas por face (Item B3)
        for idxs, _, cor in faces_ordenadas:
            poly = p[idxs]
            cv2.polylines(out, [poly], isClosed=True, color=cor, thickness=3, lineType=cv2.LINE_AA)

        # 3. Desenho dos eixos tridimensionais no canto de origem (0, 0, 0)
        p_origem = tuple(e[0])
        for k, cor_eixo in enumerate(CORES_EIXOS):
            p_eixo = tuple(e[k + 1])
            cv2.arrowedLine(out, p_origem, p_eixo, cor_eixo, 3, cv2.LINE_AA, tipLength=0.15)

        # Legenda dos eixos
        cv2.putText(out, "X", tuple(e[1] + np.array([4, 4])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "Y", tuple(e[2] + np.array([4, 4])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, "Z", tuple(e[3] + np.array([4, 4])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # 4. HUD Informativo e Legenda das Cores das Faces
        self._desenhar_hud(out, frame_idx, fps_atual, rvec, tvec, rms_reproj, jitter)
        return out, jitter

    def _desenhar_hud(
        self,
        img: np.ndarray,
        frame_idx: int,
        fps: float,
        rvec: np.ndarray,
        tvec: np.ndarray,
        rms: float,
        jitter: float,
    ) -> None:
        """Desenha painel HUD translúcido com telemetria da pose e legenda de faces."""
        h, w = img.shape[:2]

        # Painel Superior Esquerdo: Telemetria da Pose
        box_w, box_h = 420, 140
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (12 + box_w, 12 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)
        cv2.rectangle(img, (12, 12), (12 + box_w, 12 + box_h), (80, 80, 80), 1)

        cv2.putText(img, "REALIDADE AUMENTADA: POSE & CUBO 3D", (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        rv = rvec.ravel()
        tv = tvec.ravel()
        cv2.putText(img, f"Frame: {frame_idx:03d} | FPS: {fps:.1f} | Reproj: {rms:.3f} px", (22, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"rvec: [{rv[0]:+.3f}, {rv[1]:+.3f}, {rv[2]:+.3f}] rad", (22, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"tvec: [{tv[0]:+.3f}, {tv[1]:+.3f}, {tv[2]:+.3f}] m", (22, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"Estabilidade (jitter): {jitter:.3f} px/frame^2", (22, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

        # Painel Superior Direito: Legenda das Faces do Cubo
        leg_w, leg_h = 240, 160
        x0_leg = w - leg_w - 12
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (x0_leg, 12), (x0_leg + leg_w, 12 + leg_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.75, img, 0.25, 0, dst=img)
        cv2.rectangle(img, (x0_leg, 12), (x0_leg + leg_w, 12 + leg_h), (80, 80, 80), 1)

        cv2.putText(img, "CORES DAS FACES:", (x0_leg + 10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        y_offset = 52
        for _, nome_face, cor_face in FACES_CUBO:
            cv2.rectangle(img, (x0_leg + 12, y_offset - 10), (x0_leg + 28, y_offset + 2), cor_face, -1)
            cv2.rectangle(img, (x0_leg + 12, y_offset - 10), (x0_leg + 28, y_offset + 2), (255, 255, 255), 1)
            cv2.putText(img, f"Face {nome_face}", (x0_leg + 36, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            y_offset += 18


# =============================================================================
# ITEM A: CALIBRAÇÃO DE CÂMERA, DISTORÇÃO E PAINEL COMPARATIVO
# =============================================================================

def executar_item_a(
    pasta_entradas: Path = ENTRADAS_DIR,
    pasta_saidas: Path = SAIDAS_DIR,
    pattern: Tuple[int, int] = PADRAO_TABULEIRO,
    square_m: float = LADO_QUADRADO_M,
    sem_janela: bool = False,
) -> Dict[str, Any]:
    """Executa estritamente o Item A do enunciado:

    1. Carrega ao menos 15 imagens de tabuleiro de xadrez na pasta entradas.
    2. Usa cv2.findChessboardCorners e cv2.calibrateCamera para obter K, distorção e erros.
    3. Aplica cv2.undistort em imagem distorcida e gera painel comparativo lado a lado.
    4. Imprime K, os 5 coeficientes de distorção e o erro de reprojeção médio em pixels.
    5. Salva a calibração em calibracao.npz e o painel em painel_undistort.png.
    """
    print("\n" + "=" * 80)
    print("ITEM A: CALIBRAÇÃO DA CÂMERA E CORREÇÃO DE DISTORÇÃO ÓPTICA")
    print("=" * 80)

    # 1. Carregar lista de imagens
    arquivos_img = sorted(list(pasta_entradas.glob("*.png")) + list(pasta_entradas.glob("*.jpg")))
    if len(arquivos_img) < 15:
        raise RuntimeError(f"O enunciado exige ao menos 15 imagens. Encontradas apenas {len(arquivos_img)} em {pasta_entradas}.")

    print(f"[Item A] Total de imagens encontradas: {len(arquivos_img)} (mínimo exigido: 15)")
    print(f"[Item A] Padrão do tabuleiro: {pattern[0]} colunas x {pattern[1]} linhas de cantos internos")
    print(f"[Item A] Tamanho do quadrado impresso: {square_m * 1000:.1f} mm ({square_m} m)")

    # 2. Preparar pontos do objeto 3D
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    imagens_validas: List[Path] = []
    img_size: Optional[Tuple[int, int]] = None
    primeira_img_colorida: Optional[np.ndarray] = None
    primeira_img_cantos: Optional[np.ndarray] = None

    print("\n[Item A] Detectando cantos do tabuleiro e refinando com cv2.cornerSubPix:")
    for idx, p in enumerate(arquivos_img, 1):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (img_bgr.shape[1], img_bgr.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ret:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            objpoints.append(objp.copy())
            imgpoints.append(corners_sub)
            imagens_validas.append(p)
            if primeira_img_colorida is None:
                primeira_img_colorida = img_bgr.copy()
                primeira_img_cantos = corners_sub.copy()
            print(f"  [{idx:02d}/{len(arquivos_img):02d}] {p.name:<15} -> SUCESSO ({cols * rows} cantos refinados)")
        else:
            print(f"  [{idx:02d}/{len(arquivos_img):02d}] {p.name:<15} -> FALHA na detecção")

    if len(objpoints) < 15:
        raise RuntimeError(f"Detecção bem-sucedida em apenas {len(objpoints)} imagens. Mínimo exigido: 15.")

    print(f"\n[Item A] {len(objpoints)} imagens válidas para calibração. Executando cv2.calibrateCamera...")

    # 3. Calibração com cv2.calibrateCamera
    ret_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # 4. Cálculo dos erros de reprojeção por imagem e médio
    erros_por_imagem: List[float] = []
    total_pontos = 0
    total_erro_sq = 0.0

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        diff = imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)
        erro_img = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        erros_por_imagem.append(erro_img)
        total_erro_sq += float(np.sum(diff ** 2))
        total_pontos += len(objpoints[i])

    erro_reproj_medio = float(np.mean(erros_por_imagem))
    erro_rms_global = float(np.sqrt(total_erro_sq / total_pontos))

    # 5. Impressão dos resultados exigidos pelo enunciado
    print("\n" + "-" * 80)
    print("RESULTADOS DA CALIBRAÇÃO DA CÂMERA (ESTRITAMENTE EXIGIDOS PELO ENUNCIADO):")
    print("-" * 80)

    print("\n1. MATRIZ INTRÍNSECA K (3x3):")
    print(f"    [[ {K[0,0]:11.4f}, {K[0,1]:11.4f}, {K[0,2]:11.4f} ],")
    print(f"     [ {K[1,0]:11.4f}, {K[1,1]:11.4f}, {K[1,2]:11.4f} ],")
    print(f"     [ {K[2,0]:11.4f}, {K[2,1]:11.4f}, {K[2,2]:11.4f} ]]")
    print(f"    -> Distância Focal Horizontal (fx): {K[0, 0]:.4f} px")
    print(f"    -> Distância Focal Vertical   (fy): {K[1, 1]:.4f} px")
    print(f"    -> Ponto Principal Óptico (cx, cy): ({K[0, 2]:.4f}, {K[1, 2]:.4f}) px")

    d = dist.ravel()
    k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]
    print("\n2. OS 5 COEFICIENTES DE DISTORÇÃO [k1, k2, p1, p2, k3]:")
    print(f"    k1 (Radial 1)     : {k1:+11.6f}  (Distorção em Barril < 0)")
    print(f"    k2 (Radial 2)     : {k2:+11.6f}  (Ajuste radial secundário)")
    print(f"    p1 (Tangencial 1) : {p1:+11.6f}  (Desalinhamento lente-sensor)")
    print(f"    p2 (Tangencial 2) : {p2:+11.6f}  (Desalinhamento lente-sensor)")
    print(f"    k3 (Radial 3)     : {k3:+11.6f}  (Ajuste radial periférico)")

    print("\n3. ERRO DE REPROJEÇÃO MÉDIO:")
    print(f"    Erro Médio por Imagem : {erro_reproj_medio:.4f} pixels")
    print(f"    Erro RMS Global       : {erro_rms_global:.4f} pixels")
    # Classificação derivada dos limiares definidos no comentário técnico do topo do arquivo
    if erro_reproj_medio < 0.5:
        classificacao = "EXCELENTE / SUBPIXEL (MRE < 0.5 px - adequado para robótica de precisão, SLAM e AR)"
    elif erro_reproj_medio <= 1.0:
        classificacao = "BOM / ACEITÁVEL (0.5 <= MRE <= 1.0 px - adequado para navegação de robôs móveis)"
    else:
        classificacao = "INACEITÁVEL (MRE > 1.0 px - recalibrar com mais vistas, melhor iluminação e cantos nítidos)"
    print(f"    Classificação Robótica : {classificacao}")

    # Tabela de erros por imagem
    print("\n4. ERROS DE REPROJEÇÃO INDIVIDUAIS POR IMAGEM:")
    print(f"    {'Imagem':<18} | {'Erro RMS (px)':<14} | {'Status':<15}")
    print("    " + "-" * 50)
    for p_img, err in zip(imagens_validas, erros_por_imagem):
        if err < 0.1:
            status = "Excelente (<0.1px)"
        elif err < 0.5:
            status = "Bom (<0.5px)"
        elif err <= 1.0:
            status = "Aceitável (<=1.0px)"
        else:
            status = "Inaceitável (>1.0px)"
        print(f"    {p_img.name:<18} | {err:14.4f} | {status:<15}")

    # Comparação com o gabarito sintético (verdade.npz) se presente
    caminho_verdade = pasta_entradas / "verdade.npz"
    if caminho_verdade.exists():
        with np.load(str(caminho_verdade)) as vz:
            Kv = vz["K"]
            dv = vz["dist"].ravel()
        print("\n5. COMPARAÇÃO COM OS VALORES VERDADEIROS DA CÂMERA VIRTUAL (verdade.npz):")
        print(f"    {'Parâmetro':<10} | {'Verdadeiro':<12} | {'Calibrado':<12} | {'Erro Absoluto':<15} | {'Erro Relativo (%)':<15}")
        print("    " + "-" * 72)
        for nome, v_real, v_est in [("fx", Kv[0, 0], K[0, 0]), ("fy", Kv[1, 1], K[1, 1]),
                                    ("cx", Kv[0, 2], K[0, 2]), ("cy", Kv[1, 2], K[1, 2])]:
            err_abs = v_est - v_real
            err_rel = (err_abs / v_real) * 100.0
            print(f"    {nome:<10} | {v_real:12.4f} | {v_est:12.4f} | {err_abs:+15.4f} | {err_rel:+15.3f}%")
        for nome, v_real, v_est in zip(["k1", "k2", "p1", "p2", "k3"], dv, d[:5]):
            err_abs = v_est - v_real
            print(f"    {nome:<10} | {v_real:12.5f} | {v_est:12.5f} | {err_abs:+15.5f} | {'N/A':<15}")

    # 6. Aplicação de cv2.undistort e montagem de painel comparativo lado a lado
    print("\n[Item A] Aplicando cv2.undistort e gerando painel Original x Corrigida...")
    assert primeira_img_colorida is not None
    img_distorcida = primeira_img_colorida.copy()
    img_corrigida = cv2.undistort(img_distorcida, K, dist, None, K)

    # Adicionar guias visuais de retilinidade para comprovar a remoção do barril
    img_vis_orig = img_distorcida.copy()
    img_vis_corr = img_corrigida.copy()

    # Traçar linhas de retilinidade nos cantos detectados
    if primeira_img_cantos is not None:
        c_orig = primeira_img_cantos.reshape(-1, pattern[0], 2)
        # Pontos undistorted para desenhar linhas perfeitamente retas
        c_und = cv2.undistortPoints(primeira_img_cantos, K, dist, P=K).reshape(-1, pattern[0], 2)

        for row_pts in c_orig:
            cv2.polylines(img_vis_orig, [np.round(row_pts).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
        for row_pts in c_und:
            cv2.polylines(img_vis_corr, [np.round(row_pts).astype(np.int32)], False, (0, 200, 0), 2, cv2.LINE_AA)

    # Montar painel lado a lado com linha divisória branca central
    h_img, w_img = img_vis_orig.shape[:2]
    painel = np.hstack([img_vis_orig, img_vis_corr])
    cv2.line(painel, (w_img, 0), (w_img, h_img), (255, 255, 255), 3)

    # Rótulos destacados no painel
    cv2.rectangle(painel, (16, 16), (420, 64), (20, 20, 20), -1)
    cv2.rectangle(painel, (16, 16), (420, 64), (0, 0, 255), 2)
    cv2.putText(painel, "ORIGINAL (Distorcida - Curvatura Barril)", (26, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

    cv2.rectangle(painel, (w_img + 16, 16), (w_img + 450, 64), (20, 20, 20), -1)
    cv2.rectangle(painel, (w_img + 16, 16), (w_img + 450, 64), (0, 200, 0), 2)
    cv2.putText(painel, "CORRIGIDA (cv2.undistort - Retificada)", (w_img + 26, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

    # Salvar painel e calibração
    pasta_saidas.mkdir(parents=True, exist_ok=True)
    caminho_painel = pasta_saidas / "painel_undistort.png"
    cv2.imwrite(str(caminho_painel), painel)
    print(f"[Item A] Painel comparativo salvo com sucesso em: {caminho_painel}")

    caminho_calib = pasta_saidas / "calibracao.npz"
    np.savez_compressed(
        str(caminho_calib),
        K=K,
        dist=dist,
        pattern=np.array(pattern),
        square_m=np.array(square_m),
        image_size=np.array(img_size),
        erro_reproj_medio=erro_reproj_medio,
        erro_rms_global=erro_rms_global,
    )
    print(f"[Item A] Parâmetros de calibração salvos em: {caminho_calib}")

    if not sem_janela:
        cv2.imshow("Validacao Item A - Original vs Corrigida (cv2.undistort)", painel)
        print("[Item A] Exibindo janela de validação. Pressione qualquer tecla ou aguarde...")
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

    return {
        "K": K,
        "dist": dist,
        "erro_reproj_medio": erro_reproj_medio,
        "erro_rms_global": erro_rms_global,
        "erros_por_imagem": erros_por_imagem,
        "painel_path": caminho_painel,
        "calibracao_path": caminho_calib,
    }


# =============================================================================
# ITEM B: REALIDADE AUMENTADA EM TEMPO REAL COM CUBO 3D SOBREPOSTO
# =============================================================================

def executar_item_b(
    caminho_video: Path,
    caminho_calibracao: Path,
    pasta_saidas: Path = SAIDAS_DIR,
    pattern: Tuple[int, int] = PADRAO_TABULEIRO,
    square_m: float = LADO_QUADRADO_M,
    max_frames: int = 0,
    sem_janela: bool = False,
) -> Dict[str, Any]:
    """Executa estritamente o Item B do enunciado:

    1. Carrega os parâmetros de calibração obtidos no Item A e instancia a Câmera Virtual.
    2. Lê o vídeo gravado do tabuleiro da pasta entradas.
    3. Em tempo real/frame a frame:
       a) Detecta o tabuleiro com cv2.findChessboardCorners + cornerSubPix.
       b) Estima a pose 6DoF (rvec, tvec) com cv2.solvePnP.
       c) Projeta um cubo 3D virtual com aresta de 1 quadrado sobre o canto de origem com cv2.projectPoints.
       d) Desenha as arestas do cubo com cores distintas por face e preenchimento ordenado por profundidade.
       e) Exibe o vetor de rotação e translação estimados no terminal a cada frame.
    4. Grava o feed aumentado em vídeo (saidas/video_aumentado.avi) e salva frames representativos.
    5. Mede a estabilidade do cubo (métrica de jitter temporal de vértices).
    """
    print("\n" + "=" * 80)
    print("ITEM B: REALIDADE AUMENTADA EM TEMPO REAL COM CUBO 3D SOBREPOSTO")
    print("=" * 80)

    # 1. Carregar calibração do Item A
    if not caminho_calibracao.exists():
        raise FileNotFoundError(f"Arquivo de calibração não encontrado em {caminho_calibracao}. Execute o Item A primeiro.")

    with np.load(str(caminho_calibracao)) as calib:
        K = calib["K"]
        dist = calib["dist"]
        pattern = tuple(int(v) for v in calib["pattern"])
        square_m = float(calib["square_m"])
        img_size = tuple(int(v) for v in calib["image_size"])

    print(f"[Item B] Calibração carregada de: {caminho_calibracao}")
    print(f"[Item B] Padrão do tabuleiro: {pattern[0]}x{pattern[1]} | Quadrado: {square_m * 1000:.1f} mm")
    print(f"[Item B] Cubo virtual 3D com aresta de 1 quadrado = {square_m * 1000:.1f} mm ({square_m} m)")
    print(f"[Item B] Canto de origem: V0 = (0, 0, 0) no primeiro canto interno do tabuleiro")

    # 2. Instanciar a Câmera Virtual
    camera = CameraVirtual(K=K, dist=dist, pattern=pattern, square_m=square_m, image_size=img_size)

    # 3. Abrir vídeo gravado do tabuleiro
    if not caminho_video.exists():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado em {caminho_video}.")

    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o arquivo de vídeo: {caminho_video}")

    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Item B] Vídeo aberto: {caminho_video.name}")
    print(f"[Item B] Resolução: {largura}x{altura} | FPS nominal: {fps_nominal:.1f} | Total frames: {total_frames_video}")

    # 4. Configurar gravação do vídeo aumentado
    pasta_saidas.mkdir(parents=True, exist_ok=True)
    caminho_video_saida = pasta_saidas / "video_aumentado.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(caminho_video_saida), fourcc, fps_nominal, (largura, altura))

    caminho_csv = pasta_saidas / "poses_estimadas.csv"
    csv_file = open(caminho_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "tempo_s", "rx_rad", "ry_rad", "rz_rad",
        "tx_m", "ty_m", "tz_m", "reproj_rms_px", "jitter_px"
    ])

    print("\n[Item B] Processando frames e exibindo pose (rvec e tvec) no terminal a cada frame:")
    print("-" * 105)
    print(f"{'Frame':<8} | {'rvec [rx, ry, rz] (radianos)':<38} | {'tvec [tx, ty, tz] (metros)':<34} | {'Reproj':<9} | {'Jitter'}")
    print("-" * 105)

    frame_idx = 0
    frames_detectados = 0
    lista_jitters: List[float] = []
    lista_tempos: List[float] = []
    # Snapshots no primeiro frame, no frame central e no último frame realmente processado
    ultimo_frame = max(0, total_frames_video - 1)
    if max_frames > 0:
        ultimo_frame = min(ultimo_frame, max_frames - 1)
    frames_salvar = {
        0: "frame_cubo_inicial.png",
        ultimo_frame // 2: "frame_cubo_meio.png",
        ultimo_frame: "frame_cubo_final.png",
    }

    t_inicio_global = time.perf_counter()

    while True:
        t_inicio_frame = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_idx >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_corners, corners = camera.detect_chessboard(gray)

        frame_aumentado = frame.copy()
        rvec_str, tvec_str, reproj_str, jitter_str = "None", "None", "N/A", "N/A"

        if ok_corners and corners is not None:
            ok_pose, rvec, tvec, rms_reproj = camera.estimate_pose(corners)
            if ok_pose and rvec is not None and tvec is not None:
                frames_detectados += 1
                fps_atual = 1.0 / max(1e-4, time.perf_counter() - t_inicio_frame)

                # Renderizar Cubo 3D virtual com aresta de 1 quadrado sobre o canto de origem
                frame_aumentado, jitter = camera.render_ar_cube(
                    frame=frame,
                    rvec=rvec,
                    tvec=tvec,
                    aresta_m=square_m,  # aresta de 1 quadrado
                    frame_idx=frame_idx,
                    fps_atual=fps_atual,
                    rms_reproj=rms_reproj,
                )
                if jitter > 0:
                    lista_jitters.append(jitter)

                rv = rvec.ravel()
                tv = tvec.ravel()
                rvec_str = f"[{rv[0]:+7.4f}, {rv[1]:+7.4f}, {rv[2]:+7.4f}]"
                tvec_str = f"[{tv[0]:+7.4f}, {tv[1]:+7.4f}, {tv[2]:+7.4f}]"
                reproj_str = f"{rms_reproj:6.4f} px"
                jitter_str = f"{jitter:6.4f} px"

                # Gravar telemetria da pose no CSV
                csv_writer.writerow([
                    frame_idx,
                    round(frame_idx / fps_nominal, 4),
                    round(rv[0], 5), round(rv[1], 5), round(rv[2], 5),
                    round(tv[0], 5), round(tv[1], 5), round(tv[2], 5),
                    round(rms_reproj, 4), round(jitter, 4)
                ])

        # Exibição obrigatória a cada frame no terminal (Item B4)
        print(f"[{frame_idx:03d}/{total_frames_video:03d}] | rvec: {rvec_str:<32} | tvec: {tvec_str:<28} | {reproj_str:<9} | {jitter_str}")

        # Gravar frame aumentado no vídeo de saída
        writer.write(frame_aumentado)

        # Salvar snapshots específicos para auditoria/relatório
        if frame_idx in frames_salvar:
            nome_snapshot = frames_salvar[frame_idx]
            caminho_snap = pasta_saidas / nome_snapshot
            cv2.imwrite(str(caminho_snap), frame_aumentado)
            print(f"  >>> Snapshot salvo: {caminho_snap.name} <<<")

        # Visualização interativa
        if not sem_janela:
            cv2.imshow("Validacao Item B - Realidade Aumentada (Cubo 3D Estavel)", frame_aumentado)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("\n[Item B] Interrupção solicitada pelo usuário.")
                break

        lista_tempos.append(time.perf_counter() - t_inicio_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    if not sem_janela:
        cv2.destroyAllWindows()

    tempo_total = time.perf_counter() - t_inicio_global
    fps_medio_real = frame_idx / max(1e-4, tempo_total)
    jitter_medio = float(np.mean(lista_jitters)) if lista_jitters else 0.0
    jitter_max = float(np.max(lista_jitters)) if lista_jitters else 0.0

    print("-" * 105)
    print("\nRESUMO DA VALIDAÇÃO DO ITEM B:")
    print("=" * 80)
    print(f"Total de Frames Processados : {frame_idx}/{total_frames_video}")
    print(f"Frames com Pose e Cubo      : {frames_detectados}/{frame_idx} (Taxa de detecção: {frames_detectados/max(1,frame_idx)*100:.1f}%)")
    print(f"Taxa Média de Processamento : {fps_medio_real:.1f} FPS (Tempo total: {tempo_total:.2f} s)")
    print(f"Estabilidade Média (Jitter) : {jitter_medio:.4f} pixels/frame^2")
    print(f"Jitter Máximo               : {jitter_max:.4f} pixels/frame^2")
    # Veredito derivado das medições: taxa de detecção e jitter médio dos vértices projetados
    taxa_deteccao = frames_detectados / max(1, frame_idx)
    if taxa_deteccao >= 0.95 and jitter_medio < 1.0:
        comportamento = "ESTÁVEL E ADERIDO AO TABULEIRO (detecção >= 95% dos frames e jitter médio < 1 px/frame^2)"
    elif taxa_deteccao >= 0.95:
        comportamento = "ADERIDO, MAS COM TREPIDAÇÃO PERCEPTÍVEL (jitter médio >= 1 px/frame^2)"
    else:
        comportamento = "INSTÁVEL: TABULEIRO NÃO DETECTADO EM PARTE DOS FRAMES (detecção < 95%)"
    print(f"Comportamento do Cubo       : {comportamento}")
    print(f"Vídeo Aumentado Salvo       : {caminho_video_saida}")
    print(f"Tabela de Poses em CSV      : {caminho_csv}")
    print("=" * 80)

    return {
        "frames_processados": frame_idx,
        "frames_detectados": frames_detectados,
        "fps_medio": fps_medio_real,
        "jitter_medio": jitter_medio,
        "video_saida": caminho_video_saida,
        "csv_poses": caminho_csv,
    }


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL (MAIN CLI)
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execução Estrita do Enunciado do Trabalho de Visão Computacional (Exercício 1: Calibração e Realidade Aumentada)"
    )
    parser.add_argument("--item", type=str, choices=["a", "b", "ambos"], default="ambos",
                        help="Escolha o item a executar: 'a' (calibração), 'b' (realidade aumentada) ou 'ambos' (padrão)")
    parser.add_argument("--sem-janela", action="store_true",
                        help="Não abre janelas gráficas do OpenCV (ideal para execuções automáticas/background)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Número máximo de frames a processar no Item B (0 = todos os frames do vídeo)")
    parser.add_argument("--video", type=str, default=None,
                        help="Caminho do vídeo para o Item B (padrão: entradas/video_sintetico.avi)")

    args = parser.parse_args()

    ENTRADAS_DIR.mkdir(parents=True, exist_ok=True)
    SAIDAS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("### EXERCÍCIO 1: CALIBRAÇÃO DE CÂMERA E REALIDADE AUMENTADA EM TEMPO REAL ###")
    print("#" * 80)

    caminho_calib = SAIDAS_DIR / "calibracao.npz"
    caminho_video = Path(args.video) if args.video else (ENTRADAS_DIR / "video_sintetico.avi")

    t_inicio = time.perf_counter()

    if args.item in ("a", "ambos"):
        executar_item_a(
            pasta_entradas=ENTRADAS_DIR,
            pasta_saidas=SAIDAS_DIR,
            pattern=PADRAO_TABULEIRO,
            square_m=LADO_QUADRADO_M,
            sem_janela=args.sem_janela,
        )

    if args.item in ("b", "ambos"):
        # Se for rodar só o B e a calibração não estiver na saida, verifica se já existe
        if not caminho_calib.exists():
            print("\n[Aviso] Calibração não encontrada em saidas/calibracao.npz. Executando Item A primeiro...")
            executar_item_a(
                pasta_entradas=ENTRADAS_DIR,
                pasta_saidas=SAIDAS_DIR,
                pattern=PADRAO_TABULEIRO,
                square_m=LADO_QUADRADO_M,
                sem_janela=True,
            )

        executar_item_b(
            caminho_video=caminho_video,
            caminho_calibracao=caminho_calib,
            pasta_saidas=SAIDAS_DIR,
            pattern=PADRAO_TABULEIRO,
            square_m=LADO_QUADRADO_M,
            max_frames=args.max_frames,
            sem_janela=args.sem_janela,
        )

    t_total = time.perf_counter() - t_inicio
    print("\n" + "#" * 80)
    print(f"### EXECUÇÃO COMPLETA E ESTRITA DO ENUNCIADO FINALIZADA COM SUCESSO ({t_total:.2f} s) ###")
    print(f"Arquivos gerados em: {SAIDAS_DIR.resolve()}")
    print("#" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
