import os
import csv
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Caminho do modelo e da pasta com imagens
MODEL_PATH = "modeloM.pt"
IMAGES_DIR = "images"
OUTPUT_CSV = "results.csv"

# Carrega o modelo YOLO
model = YOLO(MODEL_PATH)

# Lista de imagens válidas
valid_exts = ('.jpg', '.jpeg', '.png')
image_paths = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.lower().endswith(valid_exts)]

# Detecta uma imagem apenas para obter as classes disponíveis
dummy_result = model(image_paths[0])
class_names = dummy_result[0].names  # Dicionário de classes (ex: {0: 'person', 1: 'car'})

# Cria cabeçalho do CSV
header = ["image_path"] + list(class_names.values())

# Lista para armazenar resultados
rows = []

# modelo de descrição textual
text_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
)

def describeImage(image_path: str) -> str:
    try:
        result = model.caption(image_path, length="normal")

        caption_text = result[0].caption if hasattr(result[0], "caption") else None

        if caption_text:
            return caption_text
        else:
            return "Não foi possível gerar uma descrição para esta imagem."

    except Exception as e:
        return f"Erro ao descrever a imagem: {e}"


for img_path in image_paths:
    # Executa a detecção
    results = model(img_path, verbose=False)
    result = results[0]

    # Conta quantas instâncias de cada classe foram detectadas
    counts = {cls_name: 0 for cls_name in class_names.values()}
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]
        counts[cls_name] += 1

    # Geração da descrição textual
    description = describeImage(img_path)
    
    # Monta linha para o CSV
    row = {"image_path": img_path, "description": description}
    row.update(counts)
    rows.append(row)

# Escreve os resultados no CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Detecções concluídas! Resultados salvos em '{OUTPUT_CSV}'.")
