# clear_result_folder.py
from pathlib import Path

RESULT_DIR = Path("./ExternalFiles")

if RESULT_DIR.exists() and RESULT_DIR.is_dir():
    count = 0
    for file in RESULT_DIR.iterdir():
        if file.is_file():
            file.unlink()
            count += 1
    print(f"[OK] {count} Dateien in '{RESULT_DIR}' gelöscht.")
else:
    print(f"[HINWEIS] Ordner '{RESULT_DIR}' existiert nicht oder ist kein Verzeichnis.")
