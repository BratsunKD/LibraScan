import os
import io

from loguru import logger
import hashlib
import aiohttp
import asyncio
from PIL import Image
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form

from app.model import Classifier
from app.utils import get_hash
from app.database import DB, DB_CONFIG

db = DB(DB_CONFIG)

IMAGE_ENCODER_URL = os.getenv("IMAGE_ENCODER_URL", "http://localhost:8001/encode_image")
TEXT_ENCODER_URL = os.getenv("TEXT_ENCODER_URL", "http://localhost:8002/encode_text")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.create_pool()
    app.state.db = db
    app.state.classifier = Classifier()
    yield
    await db.close_pool()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_prediction(contents: bytes, filename: str, result: int, hash_id: str):
    os.makedirs("uploads", exist_ok=True)

    # Сохранение файла
    # async with aiofiles.open(filename, "wb") as f:
    #     await f.write(contents)

    await db.save_prediction(hash_id, datetime.utcnow(), filename, result)


async def fetch_image_vector(image_bytes: bytes, filename: str, content_type: str):
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', image_bytes, filename=filename, content_type=content_type)
        resp = await session.post(IMAGE_ENCODER_URL, data=data)
        resp.raise_for_status()
        return (await resp.json())['vector']


async def fetch_text_vector(text_bytes: bytes, filename: str = "input.txt", content_type: str = "text/plain"):
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field(
            name='file',
            value=io.BytesIO(text_bytes),
            filename=filename,
            content_type=content_type
        )
        async with session.post(TEXT_ENCODER_URL, data=data) as resp:
            resp.raise_for_status()
            return (await resp.json())['vector']


@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    contents = await file.read()

    image_bytes = contents

    img_task = fetch_image_vector(image_bytes, file.filename, file.content_type)
    #txt_task = fetch_text_vector(image_bytes, file.filename, file.content_type)

    image_vec = await asyncio.gather(img_task)

    logger.info(f"image_vec={len(image_vec)}")
    # logger.info(f"text_vec={len(text_vec)}")
    result = app.state.classifier.predict(image_vec)

    file_root, file_ext = os.path.splitext(file.filename)
    hash_id = get_hash(file.filename)
    new_filename = f"{hash_id}{file_ext}"
    save_path = f"uploads/{new_filename}"

    if background_tasks:
        background_tasks.add_task(save_prediction, contents, save_path, result, hash_id)

    return {"class": result, "hash": hash_id}


@app.post("/save_answer")
async def save_answer(
    prediction_hash: str = Form(...),
    user_answer: str = Form(...)
):
    await db.save_user_answer(
        prediction_hash=prediction_hash,
        user_answer=int(user_answer), # не ок
        answered_at=datetime.utcnow()
    )
    return {"status": "success", "hash": prediction_hash}


