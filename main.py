from trainer import init
from utils.engine_decorators import start
from utils.options import args
import torch
import numpy as np
from utils.eval import eval

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


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    ##TODO eww
    #trainer, SR, D, vgg, loader, schedulerD, schedulerG, optimizerD, optimizerG, resume_epoch, resume_inter = init()
    #start(trainer, SR, D, vgg, loader, schedulerD, schedulerG, optimizerD, optimizerG, resume_epoch, resume_inter)
    #trainer.run(loader, args.epochs)

    eval()
