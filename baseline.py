import warnings
warnings.filterwarnings("ignore")

from fastai.vision import *
# from fastai.metrics import error_rate
from fastai.callbacks import *

import torch

torch.cuda.empty_cache()
defaults.device = torch.device("cuda")

def get_data(bs, size): 
    tfms = get_transforms(max_lighting=0.4, max_zoom=1.2, max_warp=0.2, max_rotate=20, xtra_tfms=[flip_lr()])
    return ImageDataBunch.from_folder(Path('./dataset'),
                                  train = 'train/',
                                  valid_pct = 0.1,
                                  resize_method=ResizeMethod.SQUISH, 
                                  ds_tfms = tfms,
                                  size = size,
                                  bs = bs,
                                  num_workers = 50
                                  ).normalize(imagenet_stats)


data = get_data(64, 299)
learn = cnn_learner(data, models.resnet50, metrics=accuracy).to_fp16()
"""
learn.lr_find()
learn.recorder.plot(suggestion=True)
mgr = learn.recorder.min_grad_lr
learn.fit_one_cycle(5, max_lr=mgr)

learn.save('baseline-1')
print('selesai baseline 1 ')
"""
learn.load('baseline-1')
learn.lr_find()
learn.recorder.plot(suggestion=True)
mgr = learn.recorder.min_grad_lr

# finetune
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(mgr/10, mgr*10))

learn.save('baseline')

