from fastapi import FastAPI, Form, File, UploadFile
import numpy as np
import os
import gdown
from train_model.text_encoder import TextEncoder

app = FastAPI()


url = "https://drive.google.com/drive/folders/1z5E0u_dPoZGU9tM7TIZbpJQV1dFgE0Bk"
model_path = "weights/text_encoder.pth"


model = TextEncoder(model_path)


@app.post("/encode_text")
async def encode_text(file: UploadFile = File(...)):
    contents = await file.read()
    vector = model.get_ocr_text_emb_from_bytes(contents)
    return {"vector": vector}
