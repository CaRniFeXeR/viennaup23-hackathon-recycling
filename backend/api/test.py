from pathlib import Path

root_folder = Path("root_folder")


for file in root_folder.iterdir():
    if file.name.startswith("pred_after_"):
        