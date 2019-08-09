import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from ignite.engine import Engine
from torch import nn

from models.SR import EDSR
from models.domain_classif import DomainCritic
from models.feature_extractor import Extractor
from models.sr_classif import SrCritic
from utils.TVLoss import TVLoss
from utils.VGG import VGG
from utils.datasetmanager import DatasetManager
from utils.options import args
from utils.utils import calculate_gradient_penalty


##TODO CHANGE DIV2K
##TODO check DPI
##TODO check encoding of pixels
##TODO try with 2 D for res and features
##TODO one gets all patches indp from domain the other indp of resolution
##TODO one D with 4 classes too


def init(DEVICES):
    feature_extractor = Extractor(1)
    domain_classifier = DomainCritic()
    resolution_classifier = DomainCritic()
    sr_classif_critic = SrCritic()
    SR = EDSR(2, 20)
    vgg = VGG()

    feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=DEVICES).cuda()
    domain_classifier = torch.nn.DataParallel(domain_classifier, device_ids=DEVICES).cuda()
    resolution_classifier = torch.nn.DataParallel(resolution_classifier, device_ids=DEVICES).cuda()
    sr_classif_critic = torch.nn.DataParallel(sr_classif_critic, device_ids=DEVICES).cuda()
    SR = torch.nn.DataParallel(SR, device_ids=DEVICES).cuda()
    vgg = torch.nn.DataParallel(vgg, device_ids=DEVICES).cuda()

    # optimizer = optim.Adam([{'params': feature_extractor.parameters()},
    #                         {'params': sr_classif_critic.parameters()},
    #                         {'params': domain_classifier.parameters()},
    #                         {'params': resolution_classifier.parameters()},
    #                         {'params': SR.parameters()}], lr=args.learning_rate, betas=(0, 0.9))

    optimizer = optim.Adam(feature_extractor.parameters(),lr=args.learning_rate)
    optimizer2 = optim.Adam(sr_classif_critic.parameters(),lr=args.learning_rate, betas=(0, 0.9))
    optimizer3 = optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    optimizer4 = optim.Adam(resolution_classifier.parameters(), lr=args.learning_rate)
    optimizer5 = optim.Adam(SR.parameters(), lr=args.learning_rate)

    dataset = DatasetManager()
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_workers, drop_last=True)
    mse = nn.MSELoss().cuda()
    domain_criterion = nn.BCELoss().cuda()
    tvloss = TVLoss().cuda()

    def step(engine, batch):
        x_source, y_source, x_target, y_target, x_source_sup, _, x_target_sup, _ = batch
        # start_steps = (engine.state.epoch - 1) * len(loader)
        # total_steps = args.epochs * len(loader)

        # p = float(engine.state.iteration + start_steps) / total_steps
        # constant = 2. / (1. + np.exp(-args.gamma * p)) - 1
        #
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.learning_rate / (1. + 10 * p) ** 0.75


        optimizer2.zero_grad()


        for p in sr_classif_critic.parameters():
            p.requires_grad = True

        for p in feature_extractor.parameters():
            p.requires_grad = False

        ##SR loss
        src_features = feature_extractor(x_source.cuda()).detach()
        SR_src = SR(src_features)
        SR_src_ = SR_src.detach()

        dloss_real = sr_classif_critic(y_source.cuda()).mean()

        dloss_fake = sr_classif_critic(SR_src_).mean()

        gp = calculate_gradient_penalty(sr_classif_critic, y_source.cuda().data, SR_src_.data)

        d_loss1 = dloss_fake - dloss_real + gp
        d_loss1.backward()
        optimizer2.step()

        for p in sr_classif_critic.parameters():
            p.requires_grad = False

        for p in feature_extractor.parameters():
            p.requires_grad = True

        optimizer.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        src_features = feature_extractor(x_source.cuda())
        tgt_features = feature_extractor(x_target.cuda())
        xsource_sup = feature_extractor(x_source_sup.cuda())
        xtarget_sup = feature_extractor(x_target_sup.cuda())
        d_loss = - sr_classif_critic(SR_src).mean() + sr_classif_critic(y_source.cuda()).mean()
        vgg_fake = vgg(SR_src)
        vgg_real = vgg(y_source.detach())
        vgg_loss = 0.01 * mse(vgg_fake, vgg_real) + d_loss + \
                   0.001 * tvloss(SR_src, y_source.cuda().detach())

        ##domain loss

        tgt_pred = domain_classifier(torch.cat((tgt_features, xtarget_sup), dim=0))
        src_pred = domain_classifier(torch.cat((src_features, xsource_sup), dim=0))
        tgt_loss = domain_criterion(tgt_pred, torch.zeros_like(tgt_pred))
        src_loss = domain_criterion(src_pred, torch.ones_like(src_pred))
        domain_loss = tgt_loss + src_loss

        ##Resolution Loss
        batch_res_sup = torch.cat((xsource_sup, xtarget_sup), dim=0)
        batch_res_down = torch.cat((tgt_features, src_features), dim=0)
        down_pred = resolution_classifier(batch_res_down)
        up_pred = resolution_classifier(batch_res_sup)
        down_loss = domain_criterion(down_pred, torch.zeros_like(down_pred))
        up_loss = domain_criterion(up_pred, torch.ones_like(up_pred))
        res_loss = down_loss + up_loss
        loss = vgg_loss + args.theta * (domain_loss + res_loss)
        loss.backward()
        optimizer.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()

        return {'tgt_loss': tgt_loss.item(),
                'src_loss': src_loss.item(),
                'vgg_loss': vgg_loss.item(),
                'loss': loss.item(),
                'dloss_real': dloss_real.item(),
                'dloss_fake': dloss_fake.item(),
                'GP': gp.item(),
                'd_loss': d_loss.item(),
                'down_loss': down_loss.item(),
                'up_loss': up_loss.item(),
                'dloss_1':d_loss1.item()
                }

    trainer = Engine(step)


    ret_objs = dict()
    ret_objs['trainer'] = trainer
    ret_objs['SR'] = SR
    ret_objs['feature_extractor'] = feature_extractor
    ret_objs['domain_classifier'] = domain_classifier
    ret_objs['resolution_classifier'] = resolution_classifier
    ret_objs['sr_classif_critic'] = sr_classif_critic
    ret_objs['optimizer'] = optimizer
    ret_objs['loader'] = loader

    return ret_objs
