import warnings
warnings.filterwarnings("ignore")

from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import *
from fastai.vision.models.efficientnet import *
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

def stage1(learn, data=None, save_filename=None, load_filname=None):
    if data: 
        learn.data=data
    learn.to_fp16()
    learn.freeze()
    learn.fit_one_cycle(5)
    
def stage2(learn, save_filename=None, ): 
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(5, slice(min_grad_lr/40, min_grad_lr))
    
def train(learn, data, save_filename, sz, bs=64, load_filename=None):
    if load_filename: 
        learn.load(load_filename, purge=True)
    data = get_data(size=sz, bs=bs)
    stage1(learn, data)
    stage2(learn)
    learn.save(save_filename)
    
model = EfficientNet.from_pretrained('efficientnet-b0')
data = get_data(64, 128)
model._fc = nn.Linear(model._fc.in_features, data.c)
learn = Learner(data, model,
               metrics = [accuracy], 
                bn_wd=False, #disable weight decay 
                loss_func = LabelSmoothingCrossEntropy(), 
                callback_fns=[BnFreeze,
                             partial(SaveModelCallback, monitor='accuracy', name='most_accurate')],
               path='./dataset').to_fp16() # because different clf layer, we use Learner.
learn.split(lambda m: (model._conv_head,))


# 128
''' UDAH SELESAI
stage1(learn)
stage2(learn)
learn.save('bs-epoch5-128')
'''
# 256
train(learn, data, 'b5-epoch5-256', bs=256, sz=64, load_filename='bs-epoch5-128')

# 384
data = get_data(64, 384)
train(learn, data, 'b5-epoch5-384',bs= 384, load_filename='b5-epoch5-256')

# 456
train(learn, data, 'b5-epoch5-456', bs=456, load_filename='b5-epoch5-384')


# 512
train(learn, data, 'b5-epoch5-512',bs= 512, load_filename='b5-epoch5-456')
