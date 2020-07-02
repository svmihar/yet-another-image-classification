import warnings
warnings.filterwarnings("ignore")

from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import *
from fastai.vision.models.efficientnet import *

from efficientnet_pytorch import EfficientNet


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

data = get_data(64, 128)
assert(len(data.classes)==42)

model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, data.c)

def get_model(pretrained=True, **kwargs): 
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, data.c) # check the top most layer
    return model 

def stage1(learn, data=None):
    if data: 
        learn.data=data
        learn.to_fp16()
        
    learn.freeze()
    learn.fit_one_cycle(5)
    
def stage2(learn, load_filename=None):
    if load_filename: 
        learn.load(load_filename, purge=True)
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(5, slice(min_grad_lr/40, min_grad_lr))

def train(learn, sz, bs, load_filename, save_filename): 
    learn.load(load_filename, purge=True)
    data = get_data(bs, sz)
    stage1(learn, data)
    stage2(learn)
    learn.save(save_filename)
    

learn = Learner(data, model,
               metrics = [accuracy], 
                bn_wd=False, #disable weight decay 
                loss_func = LabelSmoothingCrossEntropy(), 
                callback_fns=[BnFreeze,
                             partial(SaveModelCallback, monitor='accuracy', name='most_accurate')],
               path='./dataset').to_fp16() # because different clf layer, we use Learner.
learn.split(lambda m: (model._conv_head,))

'''
stage1(learn)
stage2(learn)
learn.save('b0-epoch5-128')
'''

train(learn, 256, 64, 'bs-epoch5-128', 'b0-epoch5-256')
train(learn, 384, 64, 'b0-epoch5-256', 'b0-epoch5-384')
train(learn, 468, 64, 'b0-epoch5-384', 'b0-epoch5-468')

data = get_data(16, 384)
stage1(learn, data)
stage2(learn)
learn.save('b0-epoch5-256')
learn.load('b0-epoch5-256', purge=True)


def make_submission(learn, filename):
  log_preds, test_labels = learn.get_preds(ds_type=DatasetType.Test)
  preds = np.argmax(log_preds, 1)
  a = np.array(preds)
  submission = pd.DataFrame({'image_name': os.listdir('data/test'), 'label': a})
  submission.to_csv(path/filename, index=False)

