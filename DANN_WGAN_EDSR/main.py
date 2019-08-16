import os

import numpy as np
import torch

from utils.engine_decorators import attach_decorators
from utils.eval import eval
from utils.options import args
from utils.scoring import scoring
from utils.trainer import init

DEVICES = [0, 1]

if __name__ == "__main__":
    if args.eval_epoch != 0:
        eval()
        scoring()
    else:
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
