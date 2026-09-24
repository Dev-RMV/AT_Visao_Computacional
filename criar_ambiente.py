#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
criar_ambiente.py - Cria (ou atualiza) o venv único do AT de Visão Computacional.

Uso, na raiz do projeto:
    py -3.13 criar_ambiente.py

O script cria .venv/ (se não existir) com o interpretador que o executa e instala os pacotes de
requirements.txt de cada exercício (ex1..ex4). Os quatro arquivos são idênticos; instalar todos
garante que nada falte se um deles for alterado. Rodar de novo instala apenas o que faltar.
"""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
PYTHON_VENV = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    if sys.version_info < (3, 10):
        print(f"Python {sys.version.split()[0]} é antigo demais; use Python 3.13 (o TensorFlow 2.21 tem wheel até 3.13).")
        return 1

    if not PYTHON_VENV.exists():
        print(f"[1/3] Criando o venv em {VENV} com {sys.executable}...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    else:
        print(f"[1/3] Venv já existe em {VENV}; reutilizando.")

    print("[2/3] Atualizando pip...")
    subprocess.run([str(PYTHON_VENV), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    requisitos = sorted(RAIZ.glob("ex*/requirements.txt"))
    if not requisitos:
        print("Nenhum ex*/requirements.txt encontrado.")
        return 1
    print(f"[3/3] Instalando pacotes de {len(requisitos)} arquivo(s) de requisitos...")
    for req in requisitos:
        print(f"      -> {req.relative_to(RAIZ)}")
        subprocess.run([str(PYTHON_VENV), "-m", "pip", "install", "-r", str(req)], check=True)

    print("\nAmbiente pronto. Ative com:")
    print(r"    .venv\Scripts\activate" if sys.platform == "win32" else "    source .venv/bin/activate")
    print("e execute, por exemplo:  cd ex1 && python main.py --sem-janela")
    return 0


if __name__ == "__main__":
    sys.exit(main())
