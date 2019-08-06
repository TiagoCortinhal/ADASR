import numpy as np
import torch
import os
from utils.trainer import init
from utils.engine_decorators import attach_decorators
from utils.options import args
DEVICES = [0, 1]


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].replace("\n", ",").strip()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    objects = init(DEVICES)
    attach_decorators(**objects)
    objects['trainer'].run(objects['loader'], args.epochs)
