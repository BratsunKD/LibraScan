from dotenv import load_dotenv
import os
from datetime import datetime
import asyncpg


load_dotenv()


DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
}


class DB:
    def __init__(self, config: dict):
        self.config = config
        self._pool = None

    async def create_pool(self):
        self._pool = await asyncpg.create_pool(**self.config)

    async def close_pool(self):
        if self._pool:
            await self._pool.close()

    async def save_prediction(
        self,
        hash_id: str,
        date: datetime,
        img_url: str,
        model_answer: int
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO predictions (hash, created_at, img_url, model_answer)
                VALUES ($1, $2, $3, $4)
            """, hash_id, date, img_url, model_answer)

    async def save_user_answer(
            self,
            prediction_hash: str,
            user_answer: int,
            answered_at: datetime,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_answers 
                    (prediction_hash, user_answer, answered_at)
                VALUES ($1, $2, $3)
                """,
                prediction_hash,
                user_answer,
                answered_at,
            )
