from fastai.vision import load_learner, image, torch, open_image
# from fastai.vision.models.efficientnet import *
from pathlib import Path
import numpy as np
import PIL

"""
model_path = Path("./models")
if model_path.is_dir():
    learn = load_learner("./models")
else:
    raise FileNotFoundError("no model folder found and export.pkl found")
"""

def load_image(image_file):
    pil_img = PIL.Image.open(image_file)

    img = pil_img.convert('RGB')
    img = image.pil2tensor(img, np.float32).div_(255)
    img = image.Image(img)
    return img, np.asarray(pil_img)


def get_prediction(image_tensor):
    hasil = model.predict(image_tensor)
    return str(hasil[0]), torch.max(hasil[2]).item()*100 
