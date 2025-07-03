import torch
import torchvision.transforms as transforms
from train_model.predictor import Predictor

from loguru import logger

class Classifier:
    def __init__(self):
        self.model = Predictor(use_text=False)

    def predict(self, image_vec):
        #text_tensor = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0)  # [1, 768]
        image_tensor = torch.tensor(image_vec, dtype=torch.float32).unsqueeze(0)  # [1, 2048]

        preds = self.model.predict_from_embedding(image_tensor)
        logger.info(f"{preds=}")

        pred_class = preds.argmax(dim=1).item()  # item() преобразует из тензора в int

        return pred_class

