# -*- coding:utf-8 -*-
from scipy.io import loadmat
import os
from PIL import Image
import torch
from torchvision import transforms
from scipy import stats
import numpy as np


def get_rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_xy(path, gt, transform=None):

    text = os.listdir(os.path.join(path, "text"))
    pic = os.listdir(os.path.join(path, "pic"))
    mat = []

    for te in text:
        img = pil_loader(os.path.join(path, "text", te))
        if transform is not None:
            img = transform(img)
        mat.append(img)
    for p in pic:
        img = pil_loader(os.path.join(path, "pic", p))
        if transform is not None:
            img = transform(img)
        mat.append(img)

    name = path.split("\\")[-1].split("_")
    row = (int(name[1]) - 1) * 7 + int(name[2]) - 1
    col = name[0].replace("cim", "")
    col = int(col) - 1

    img = torch.stack(mat, dim=0)

    return img, gt[row][col]


def cony(model, img_dir, mat_path):
    model = model.cuda()
    model.eval()
    pscores = []
    tscores = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dmos = loadmat(mat_path)
    gt = dmos["DMOS"]
    patch = os.listdir(img_dir)
    predicted = {}

    for i, pa in enumerate(patch):
        x, y = get_xy(os.path.join(img_dir, pa), gt, transform)
        x = x.cuda()
        x = x.unsqueeze(0)
        score = model(x)
        score = score.cpu().tolist()
        pscores = pscores + score
        tscores = tscores + [y]
        predicted[pa] = score

    srcc, _ = stats.spearmanr(pscores, tscores)
    plcc, _ = stats.pearsonr(pscores, tscores)
    rmse = get_rmse(pscores, tscores)

    return srcc, plcc, rmse, predicted
