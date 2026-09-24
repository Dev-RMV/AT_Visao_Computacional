#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exercicio04b.py - Execução do Item B do Exercício 4 (Relatório Integrativo do pipeline de percepção)

O relatório usa as latências medidas no Item A; se o Item A ainda não tiver sido executado nesta
chamada, ele é executado sem janela para obter os números.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import gerar_relatorio_integrativo, CAMINHO_RELATORIO


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exercício 4 - Item B: Relatório Integrativo")
    parser.parse_args()

    resultado = gerar_relatorio_integrativo(CAMINHO_RELATORIO)
    print(f"Resultado: {resultado['status']} ({resultado['palavras_texto']} palavras de texto puro)")


if __name__ == "__main__":
    sys.exit(main())
