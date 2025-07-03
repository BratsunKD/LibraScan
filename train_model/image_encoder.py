import os
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

# Устройство
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# class ImageEncoder:
#     def __init__(self, weights_path):
#         self.model = resnet50(pretrained=True)
#         self.model.fc = torch.nn.Identity()

#         self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

#         self.model.to(DEVICE)
#         self.model.eval()

#     def get_image_emb(self, images):
#         with torch.no_grad():
#             return self.model(images)

class ImageEncoder:
    def __init__(self, weights_path=None):
        # Загружаем модель ResNet с предобученными весами (если weights_path = None, используем ImageNet)
        self.model = resnet50(pretrained=True)
        
        # Меняем последний слой на Identity, чтобы извлекать признаки
        self.model.fc = torch.nn.Identity()

        # Если указан путь к кастомным весам, загружаем их
        if weights_path and os.path.exists(weights_path):
            print(f"Loading custom weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        else:
            print("Using pre-trained weights from ImageNet.")
        
        # Переводим модель в режим оценки
        self.model.to(DEVICE)
        self.model.eval()

    def get_image_emb(self, images):
        with torch.no_grad():
            return self.model(images)
