import torch

# Пути к данным
TRAIN_IMAGES_DIR = 'src/train'
TRAIN_LABELS_FILE = 'src/train_labels.txt'

AUGMENTED_TRAIN_IMAGES_DIR = 'src/augmented_train'
AUGMENTED_TRAIN_LABELS_FILE = 'src/augmented_train_labels.txt'

TEST_IMAGES_DIR = 'src/test'
TEST_LABELS_FILE = 'src/test_labels.txt'

# Пути к кэшу OCR текстов
OCR_CACHE_PATH = 'cache/ocr_cache.json'
OCR_CACHE_TEST_PATH = 'cache/ocr_cache_test.json'

# Пути к весам моделей
IMAGE_ENCODER_WEIGHTS_PATH = 'weights/image_encoder.pth'
TEXT_ENCODER_WEIGHTS_PATH = 'weights/text_encoder.pth'
FNN_WEIGHTS_PATH = 'weights/fnn.pth'

# Гиперпараметры
BATCH_SIZE = 8
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 3e-3

# Устройство
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Пути сохранения
MODEL_SAVE_PATH = 'saved_models/'
RESULTS_SAVE_PATH = 'results/'

# Размер скрытого слоя FNN (перенёс сюда по твоей просьбе)
FNN_HIDDEN_DIM = 1024

BEST_FNN_WEIGHTS_PATH = "saved_models/best_model_text_False_image_True.pth"
