import hashlib
from datetime import datetime


LEN_HASH = 8


def get_hash(filename: str, lenght: int = LEN_HASH) -> str:
    now_str = datetime.utcnow().isoformat()
    base_str = now_str + filename
    return hashlib.sha256(base_str.encode()).hexdigest()[:lenght]