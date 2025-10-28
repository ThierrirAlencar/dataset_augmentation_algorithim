import os
import csv
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from dotenv import load_dotenv
import torch

# === CONFIGURAÇÕES ===
load_dotenv()

MODEL_PATH = os.path.expanduser(os.getenv("MODEL_DIR"))
IMAGES_DIR = os.path.expanduser(os.getenv("IMAGE_DIR"))
OUTPUT_CSV = os.path.expanduser(os.getenv("CSV_DIR", "detection_results.csv"))

# === MODELO DE DETECÇÃO (YOLO) ===
model = YOLO(MODEL_PATH)

valid_exts = ('.jpg', '.jpeg', '.png')
image_paths = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.lower().endswith(valid_exts)]

# Detecta uma imagem apenas para obter as classes disponíveis
dummy_result = model(image_paths[0])
class_names = dummy_result[0].names

# Cria cabeçalho do CSV
header = ["image_path", "description"] + list(class_names.values())

# === MODELO DE DESCRIÇÃO TEXTUAL (MOONDREAM2) ===
model_id = "vikhyatk/moondream2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
text_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
text_model.to(device)

def describeImage(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        description = text_model.caption(image)
        return description or "Não foi possível gerar uma descrição."
    except Exception as e:
        return f"Erro ao descrever a imagem: {e}"

# === EXECUÇÃO ===
rows = []
for img_path in image_paths:
    results = model(img_path, verbose=False)
    result = results[0]

    counts = {cls_name: 0 for cls_name in class_names.values()}
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]
        counts[cls_name] += 1

    description = describeImage(img_path)
    print(f"processing: {img_path} -> {len(result.boxes)} objects -> description: {str(description)[:60]}...")

    row = {"image_path": img_path, "description": description}
    row.update(counts)
    rows.append(row)

# === SALVAMENTO ===
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Detecções concluídas! Resultados salvos em '{OUTPUT_CSV}'.")
