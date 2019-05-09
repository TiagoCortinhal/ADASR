import numpy as np
import torch
import os
from trainer import init
from utils.engine_decorators import attach_decorators
from utils.eval import eval
from utils.options import args

##TODO test this with RSx2
##TODO patch encoder adversarial
##TODO test 5 5 one SR system not 5
##TODO 1/8 1/4 1/2 <- train test will be res 1:1
##TODO smaller EDSR
##TODO encoder? with constraint
##TODO connect discrim inside the ESDR for res indep?
##TODO AE before EDSR?  and D after AE
##TODO unsupervised?  with cycle gan
##TODO one cycle to SR?! 1/2 -> 1 -> 1/2 and test

##TODO one model for each 1/8 1/4 1/2, another that 1/8 -> 1/2 WITH VGG/or others
##TODO start with AE

##TODO have patches from (x/8 and x/4) and (x/4 and x/2) on the same generator
##TODO discriminator SSIM PNSR AND MSE
##TODO SAME PATCHES downsample them to have the same ones -> to compare with different patches instead of upsampling

##TODO LR DECAY ?? ---> GRADIENT PENALTY


##TODO manage train/eval, perhaps new decorated function to perform eval at end of each epoch?
if __name__ == '__main__':

    ##QARMA Cluster returns "x1\nx2" where x1 and x2 are the id of the cuda devices for your job.
    ## Pytorch expects "x1,x2" otherwise it will only pick the x1.
    ##This corrects this problem
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].replace("\n", ",").strip()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    objects = init()
    attach_decorators(**objects)
    objects['trainer'].run(objects['loader'], args.epochs)

    #eval()
