from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import numpy as np
import gdown
import torch
from train_model.image_encoder import ImageEncoder

from loguru import logger

app = FastAPI()


# url = "https://drive.google.com/drive/folders/1z5E0u_dPoZGU9tM7TIZbpJQV1dFgE0Bk"
# model_path = "weights/image_encoder.pth"
# class ImageModel:
#     def __init__(self):
#         self.model = None
#
#     def load_weights(self, url: str, model_path: str):
#         if not os.path.exists(model_path):
#             os.makedirs(os.path.dirname(model_path), exist_ok=True)
#             gdown.download(url, model_path, quiet=False)
#         self.model = ImageEncoder()
#
#     def encode(self, image: Image.Image) -> list:
#         return np.ones(128).tolist()


model = ImageEncoder()
from torchvision import transforms

# Создаём трансформации
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Преобразует PIL → Tensor [3, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = ImageEncoder()

@app.post("/encode_image")
async def encode_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Преобразуем изображение и добавляем батч
    image_tensor = preprocess(image).unsqueeze(0)  # → [1, 3, 224, 224]

    # Получаем эмбеддинг
    with torch.no_grad():
        result = model.get_image_emb(image_tensor)

    logger.info(f"{result}")
    return {"vector": result.squeeze(0).tolist()}
