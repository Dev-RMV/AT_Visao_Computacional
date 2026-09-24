#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio02b.py - Execução do Item B do Exercício 2 (Pipeline Integrado de Percepção)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from main import (
    executar_item_b,
    gerar_notebook_sequencial,
    CAMINHO_VIDEO_BOLA,
    CAMINHO_CALIBRACAO,
    CAMINHO_TFLITE,
    FRAME_PADRAO,
    SAIDAS_DIR,
    RAIZ,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 2 - Item B: Pipeline Integrado de Percepção")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela do OpenCV")
    parser.add_argument("--frame", type=int, default=FRAME_PADRAO,
                        help=f"Índice do frame de bola.mp4 a processar (padrão: {FRAME_PADRAO}, bola inteira no quadro)")
    args = parser.parse_args()

    executar_item_b(
        caminho_video=CAMINHO_VIDEO_BOLA,
        caminho_calibracao=CAMINHO_CALIBRACAO,
        caminho_modelo=CAMINHO_TFLITE,
        pasta_saidas=SAIDAS_DIR,
        frame_alvo_idx=args.frame,
        sem_janela=args.sem_janela,
    )

    # Gera e executa (nbclient) o notebook com o mesmo frame
    gerar_notebook_sequencial(RAIZ / "exercicio02b.ipynb", frame_alvo_idx=args.frame)


if __name__ == "__main__":
    sys.exit(main())
