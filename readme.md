# Shopee Code League 2020 - Product Detection

This is source code for unsubmitted entry, that got ~.91 accuracy [[Student] Shopee Code League 2020 - Product Detection](https://www.kaggle.com/c/shopee-product-detection-student). 

## Warning

* There's no reproducibility guarantee for notebook which uses GPU and TPU
* Although we use The Unlicense, dataset and generated dataset falls under Shopee Terms and Conditions which can be seen on [Google Docs](https://docs.google.com/document/d/17mWGXdK8kW9wMxiAPWrn_1MnDtCRKxRdiSoz1u5RRDw), [Google Docs (2)](https://docs.google.com/document/d/13-ZxPKqHL0o5CG8YJSHjNe_cJUQnxjctCBRfu_S3sVc/) or [Internet Archive](https://web.archive.org/web/20200704093857/https://docs.google.com/document/d/17mWGXdK8kW9wMxiAPWrn_1MnDtCRKxRdiSoz1u5RRDw/edit)

## pre requisite
pip install fastai efficientnet_pytorch

# dataset 
[shopee classification](https://drive.google.com/uc\?id\=1CAUCMeDcrguVpfc72angE-8jW8BETwyl)

# models
## resnext 
- [x] training 
- [x] infer
### 50-32x4d
- [x] 128
- [x] 224 
- [x] 299 cuman dapet 82an

### 50-32x4d with more less augmentation + mixup and fp16
- [x] 128 training phase 1 # removed mixup karena malah bikin kaacau, pake mixup di akhir 299 aja biar lebih cepet training nya converge
- [x] 224
- [ ] 299 one epoch become 7 hours, which is ridiculuous, even with 8 batch_size
final score: ~.82

## efficientnetB3
gak ada image untuk di validate. `train_test_split` di train aja
- [x] training
- [x] ganti imagedatagenerator dari from folder ke [from dataframe](https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_dataframe/) biar lebih mudah di reproduce
### hasil 
jelek banget ini yang b3, cuman 77

## efficientnetb0
- [x] ganti validation steps ke 400
- [x] progressive training
- [ ] fastinference
final score .91


# visualize
 - [ ] [grad cam, to know which efnet are focused on](https://www.youtube.com/watch?v=gMUgyy-BlmM)
