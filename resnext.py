import torch.nn.functional as F
from torch import nn
import pretrainedmodels
import os
import logging
import warnings
from fastai.vision import *
from fastai.metrics import error_rate


from data_loader import get_data
from model_loader import get_learner


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train(
    resolution, batch_size, model_filename, model_checkpoint=None, epoch=(12, 12)
):
    """
        karena ada dua fase training, jadi epoch dijadikan set
    """
    logging.info("getting data")
    data = get_data(resolution=resolution, batch_size=batch_size)
    assert len(data.classes) == 42
    logging.info(f"got {len(data.classes)} class")
    if model_checkpoint:
        logging.info(f"upres data to {resolution}")
        learn = get_learner(data, model_checkpoint=model_checkpoint)
    else:
        logging.info("getting learner")
        learn = get_learner(data)
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    print(min_grad_lr)
    logging.info("training")
    # ------------------PHASE 1--------------------
    learn.fit_one_cycle(epoch[0], max_lr=slice(min_grad_lr / 10, min_grad_lr))
    learn.save(f"{model_filename}_1_{resolution}_fp16")
    logging.info("phase 1 done")
    learn.unfreeze()
    learn = learn.clip_grad()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    print(min_grad_lr)
    logging.info("phase 2 start")
    # ------------------PHASE 2--------------------
    learn.load(f"{model_filename}_1_{resolution}_fp16")
    learn.unfreeze()
    learn.to_fp16()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(epoch[1], slice(min_grad_lr / 10, min_grad_lr))
    learn.save(f"{model_filename}_2_{resolution}_fp16")
    logging.info("phase 2 finish")


# train(128, batch_size=64, model_filename="seresnext50_32x4d")
# -------------------------------------224-------------------------------------------
train(
    224,
    batch_size=64,
    model_filename="seresnext50_32x4d",
    model_checkpoint="seresnext50_32x4d_2_128_fp16",
    epoch=(10, 10),
)
# -------------------------------------299-------------------------------------------
train(
    299,
    batch_size=64,
    model_filename="seresnext50_32x4",
    model_checkpoint="seresnext50_32x4d_2_224_fp16",
    epoch=(6, 6),
)
