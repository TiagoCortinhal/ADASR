import os
import re

import torch
import torch.optim as optim
import torch.utils.data as data
from ignite.engine import Engine
from torch import nn
from ignite.handlers import Timer
from Critic import Critic
from EDSR import EDSR
from VGG import VGG
from utils.DatasetManager import DatasetManager2 as DatasetManager
from utils.options import args
from utils.utils import weights_init, calculate_gradient_penalty
from torch.optim.lr_scheduler import StepLR

def init():
    SR = EDSR(2, 16)
    D = Critic()
    vgg = VGG()
    vgg = torch.nn.DataParallel(vgg, device_ids=[0,1]).cuda()
    SR = torch.nn.DataParallel(SR, device_ids=[0,1]).cuda()
    D = torch.nn.DataParallel(D, device_ids=[0,1]).cuda()

    dataset = DatasetManager()
    criterion = nn.MSELoss()
    optimizerG = optim.Adam(SR.parameters(), lr=args.learning_rate, betas=(0, 0.9))
    optimizerD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0, 0.9))

    schedulerD = StepLR(optimizerG, step_size=2, gamma=0.1)
    schedulerG = StepLR(optimizerG, step_size=2, gamma=0.1)

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_workers, pin_memory=True, drop_last=True)
    mone = torch.FloatTensor([-1])
    mone = mone.cuda()

    def step(engine, batch):
        px, py = batch
        px = px.cuda()
        py = py.cuda()

        for p in D.parameters():
            p.requires_grad = True

        D.zero_grad()
        #optimizerD.zero_grad()

        fake = SR(px)
        fake_detach = fake.detach()

        dloss_real = D(py)
        dloss_real = dloss_real.mean()
        dloss_real.backward(mone, retain_graph=True)

        dloss_fake = D(fake_detach)
        dloss_fake = dloss_fake.mean()
        dloss_fake.backward(-1*mone, retain_graph=True)

        gp = calculate_gradient_penalty(D, py.data, fake_detach.data)
        gp.backward()

        d_loss = dloss_fake - dloss_real + gp
        #d_loss.backward(retain_graph=True)
        WD = dloss_real - dloss_fake
        optimizerD.step()

        for p in D.parameters():
            p.requires_grad = False

        SR.zero_grad()
        #optimizerG.zero_grad()


        gloss = D(fake)
        gloss = gloss.mean()
        gloss.backward(mone,retain_graph=True)
        vgg_fake = vgg(fake)
        vgg_real = vgg(py)
        vgg_loss = criterion(vgg_fake,vgg_real)
        vgg_loss = vgg_loss.mean()
        vgg_loss.backward()
        optimizerG.step()


        return {
            'dloss_real': dloss_real.item(),
            'dloss_fake': dloss_fake.item(),
            'GP': gp.mean().item(),
            'd_loss': d_loss.mean().item(),
            'WD': WD.mean().item(),
            'VGG': vgg_loss.item(),
            'gloss': -gloss.item()
        }

    trainer = Engine(step)
    res_epoch = 0
    res_iter = 0
    if args.c:
        files = os.listdir(args.output_dir)
        files = [name for name in files if name.endswith('.pth')]
        if len(files) > 0:
            ep = [int(re.search('.(\d*)_(\d)\.pth', ep).group(2)) for ep in files]
            iter = [int(re.search('.(\d*)_(\d)\.pth', ep).group(1)) for ep in files]
            SR = torch.load(args.output_dir + '/training_netG_{}_{}.pth'.format(max(iter),max(ep)))
            D = torch.load(args.output_dir + '/training_netD_{}_{}.pth'.format(max(iter),max(ep)))
            schedulerD = torch.load(args.output_dir + '/training_sched_D_{}_{}.pth'.format(max(iter),max(ep)))
            schedulerG = torch.load(args.output_dir + '/training_sched_G_{}_{}.pth'.format(max(iter),max(ep)))
            optimizerD = torch.load(args.output_dir + '/training_optim_D_{}_{}.pth'.format(max(iter),max(ep)))
            optimizerG = torch.load(args.output_dir + '/training_optim_G_{}_{}.pth'.format(max(iter),max(ep)))
            res_epoch = max(ep)-1
            res_iter = max(iter)-1
    if args.saved_SR:
        SR = torch.load(args.saved_SR)

    if args.saved_D:
        D = torch.load(args.saved_D)

    if args.saved_vgg:
        vgg = torch.load(args.saved_vgg)

    return trainer, SR, D, vgg, loader,schedulerD,schedulerG, optimizerG, optimizerD,res_epoch,res_iter
