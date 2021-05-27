# -*- coding:utf-8 -*-

import torch
import net
import consitency
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/best.pkl", help="input model path")
    parser.add_argument("--patch", default="../img/patch", help="input patch path")
    parser.add_argument("--dmos", default="../img/dmos/DMOS.mat", help="input dmos path")
    args = parser.parse_args()

    model_path = args.model
    patch_path = args.patch
    dmos_path = args.dmos

    model = net.Net()
    print(model)

    model.load_state_dict(torch.load(model_path))
    srcc, plcc, rmse, score = consitency.cony(
        model, patch_path, dmos_path
    )

    print("score:{}".format(score))
    print("plcc:{}, srcc:{}, rmse:{}".format(plcc, srcc, rmse))


if __name__ == "__main__":
    main()
