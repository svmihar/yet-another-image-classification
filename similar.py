#!/usr/bin/env python
# coding: utf-8

# # langsung dari library efficientnet

# In[ ]:


# list all train images + labels according to folders -> df
from tqdm.auto import tqdm
import pandas as pd
from fastai.vision import *

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

data = get_data(32, 299)
img_path = [str(x) for x in list(data.train_ds.items)] + list(data.valid_ds.items)
labels = [data.classes[x] for x in list(data.train_ds.y.items) + list(data.valid_ds.y.items)]
del data


# In[39]:


from PIL import Image
import torch 
from torchvision import transforms

from efficientnet_pytorch import EfficientNet



model = EfficientNet.from_pretrained('efficientnet-b0')
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # pake punya imagenet


# In[ ]:


def get_embeddings(img):
    img = tfms(Image.open(img)).unsqueeze(0)
    f = model.extract_features(img)
    return f[0].detach().cpu().numpy()

img_vectors = [get_embeddings(x) for x in tqdm(img_path)]

np.save('training_efnet_vectors_229', img_vectors)

'''
# In[40]:


img = tfms(Image.open('./test/56d6e2b4ccdaf0fc97f24d5aba7ad672.jpg')).unsqueeze(0)
print(img.shape)


# In[6]:


features = model.extract_features(img)


# In[8]:


features.shape # get all 1280 dim


# In[12]:


img_vec = features[0].detach().cpu().numpy()


# In[13]:


img_vec.shape


# In[20]:


len(img_vec)


# In[21]:


img_vec[0][0]
'''
