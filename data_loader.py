from fastai.vision import *
import torch

defaults.device = torch.device("cuda")


def get_data(path=Path("dataset/"), resolution=128, batch_size=64):
    if resolution < 128:
        tfms = get_transforms(
            max_lighting=0.4,
            max_zoom=1.2,
            max_warp=0.2,
            max_rotate=20,
            xtra_tfms=[
                rand_crop(),
                rand_zoom(1, 1.5),
                symmetric_warp(magnitude=(-0.2, 0.2)),
            ],
        )
    else:
        cutout_frac = 0.20
        p_cutout = 0.75
        cutout_sz = round(resolution * cutout_frac)
        cutout_tfm = cutout(n_holes=(1, 1), length=(cutout_sz, cutout_sz), p=p_cutout)
        tfms = get_transforms(
            do_flip=True,
            max_rotate=15,
            flip_vert=False,
            max_lighting=0.1,
            max_zoom=1.05,
            max_warp=0.0,
            xtra_tfms=[
                rand_crop(),
                rand_zoom(1, 1.5),
                symmetric_warp(magnitude=(-0.2, 0.2)),
                cutout_tfm,
            ],
        )
    data = ImageDataBunch.from_folder(
        path,
        train="train/",
        valid_pct=0.2,
        ds_tfms=tfms,
        size=resolution,
        bs=batch_size,
        num_workers=50,
    ).normalize(imagenet_stats)
    return data
