pip install tensorflow-gpu==1.14.0 keras==2.3.0

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
- [x] 128 training phase 1
- [ ] 224
- [ ] 299

## efficientnetB3
gak ada image untuk di validate. `train_test_split` di train aja
- [x] training
- [x] ganti imagedatagenerator dari from folder ke [from dataframe](https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_dataframe/) biar lebih mudah di reproduce
### hasil 
jelek banget ini yang b3, cuman 55S

## efficientnetb0
- [x] ganti validation steps ke 400
- [ ] training


# visualize
 - [ ] [dari sini](https://www.youtube.com/watch?v=gMUgyy-BlmM)
