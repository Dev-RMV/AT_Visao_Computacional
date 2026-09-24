#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio01b.py - Execução do Item B do Exercício 1 (Realidade Aumentada e Cubo 3D com solvePnP)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from main import (
    executar_item_a,
    executar_item_b,
    PADRAO_TABULEIRO,
    LADO_QUADRADO_M,
    ENTRADAS_DIR,
    SAIDAS_DIR,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 1 - Item B: Realidade Aumentada com Cubo 3D")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela do OpenCV")
    parser.add_argument("--max-frames", type=int, default=0, help="Máximo de frames a processar (0 = todos)")
    parser.add_argument("--video", type=str, default=None, help="Caminho do vídeo de entrada")
    args = parser.parse_args()

    caminho_calib = SAIDAS_DIR / "calibracao.npz"
    if not caminho_calib.exists():
        print("[Item B] Calibração não encontrada. Executando Item A primeiro...")
        executar_item_a(
            pasta_entradas=ENTRADAS_DIR,
            pasta_saidas=SAIDAS_DIR,
            pattern=PADRAO_TABULEIRO,
            square_m=LADO_QUADRADO_M,
            sem_janela=True,
        )

    caminho_video = Path(args.video) if args.video else (ENTRADAS_DIR / "video_sintetico.avi")
    executar_item_b(
        caminho_video=caminho_video,
        caminho_calibracao=caminho_calib,
        pasta_saidas=SAIDAS_DIR,
        pattern=PADRAO_TABULEIRO,
        square_m=LADO_QUADRADO_M,
        max_frames=args.max_frames,
        sem_janela=args.sem_janela,
    )


if __name__ == "__main__":
    sys.exit(main())
