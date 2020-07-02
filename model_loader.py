import pretrainedmodels
from torch import nn, cuda
import torch.nn.functional as F
from fastai.vision import *

cuda.empty_cache()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
        return F_loss.mean()


# model nyta buat cnn_learner
def resnext50_32x4d(pretrained=False):
    pretrained = "imagenet" if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))


def get_learner(data, model_checkpoint=None):
    learn = cnn_learner(
        data, resnext50_32x4d, pretrained=True, cut=-2, metrics=[accuracy]
    )
    learn.loss_fn = FocalLoss()
    learn = learn.to_fp16()
    if model_checkpoint is not None:
        learn.data = data
        learn.load(model_checkpoint)
        learn.freeze()
        return learn

    return learn


def resume_train(learner, data):
    pass
