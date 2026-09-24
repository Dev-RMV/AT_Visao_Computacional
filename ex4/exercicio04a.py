#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio04a.py - Execução do Item A do Exercício 4 (Segmentação Semântica DeepLabV3 vs HSV do TP1)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import executar_item_a, ENTRADAS_DIR, SAIDAS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 4 - Item A: Segmentação Semântica vs HSV")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela do OpenCV")
    args = parser.parse_args()

    executar_item_a(
        pasta_entradas=ENTRADAS_DIR,
        pasta_saidas=SAIDAS_DIR,
        sem_janela=args.sem_janela,
    )


if __name__ == "__main__":
    sys.exit(main())
