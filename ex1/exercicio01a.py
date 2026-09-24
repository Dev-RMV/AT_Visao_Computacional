#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio01a.py - Execução do Item A do Exercício 1 (Calibração e Distorção)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from main import executar_item_a, PADRAO_TABULEIRO, LADO_QUADRADO_M, ENTRADAS_DIR, SAIDAS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 1 - Item A: Calibração de Câmera")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela do OpenCV")
    args = parser.parse_args()

    executar_item_a(
        pasta_entradas=ENTRADAS_DIR,
        pasta_saidas=SAIDAS_DIR,
        pattern=PADRAO_TABULEIRO,
        square_m=LADO_QUADRADO_M,
        sem_janela=args.sem_janela,
    )


if __name__ == "__main__":
    sys.exit(main())
