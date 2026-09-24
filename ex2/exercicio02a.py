#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio02a.py - Execução do Item A do Exercício 2 (OpenCV DNN vs Keras com MobileNetV2)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from main import executar_item_a, PASTA_IMAGENS_TESTE, SAIDAS_DIR, CAMINHO_TFLITE


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 2 - Item A: OpenCV DNN vs Keras")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela do OpenCV")
    parser.add_argument("--loops", type=int, default=20, help="Número de repetições no benchmark")
    args = parser.parse_args()

    executar_item_a(
        pasta_imagens=PASTA_IMAGENS_TESTE,
        pasta_saidas=SAIDAS_DIR,
        caminho_modelo=CAMINHO_TFLITE,
        loops_benchmark=args.loops,
        sem_janela=args.sem_janela,
    )


if __name__ == "__main__":
    sys.exit(main())
