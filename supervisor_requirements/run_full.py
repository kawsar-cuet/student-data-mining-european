# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Run only the AdaBoost part
import subprocess
result = subprocess.run([
    "D:/MS program/Final Thesis/Final Thesis project/.venv/Scripts/python.exe",
    "11_explainable_ai_all_models.py"
], capture_output=False, text=True)
