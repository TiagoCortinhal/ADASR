import os
import warnings

import torchvision.utils as vutils
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from utils.checkpoint import ModelCheckpoint
from utils.options import args

PRINT_FREQ = 200
targetX_IMG_FNAME = '{:04d}_{:04d}_targetX.png'
targetY_IMG_FNAME = '{:04d}_{:04d}_targetY.png'
sourceY_IMG_FNAME = '{:04d}_{:04d}_sourceY.png'
sourceX_IMG_FNAME = '{:04d}_{:04d}_sourceX.png'
predtgt_IMG_FNAME = '{:04d}_{:04d}_predtgt.png'
predsrc_IMG_FNAME = '{:04d}_{:04d}_predsrc.png'
LOGS_FNAME = 'logs.tsv'
PLOT_FNAME = 'plot.svg'
CKPT_PREFIX = 'networks'


def attach_decorators(trainer, SR, feature_extractor,
                      domain_classifier,
                      resolution_classifier, sr_classif_critic,
                      optimizer, loader):
    timer = Timer(average=True)

    checkpoint_handler = ModelCheckpoint(args.output_dir + '/checkpoints/domain_adaptation_training/', 'training',
                                         save_interval=1, n_saved=300, require_empty=False,iteration=args.epoch_c)

    monitoring_metrics = ['tgt_loss', 'src_loss', 'vgg_loss', 'loss', 'GP', 'd_loss', 'down_loss', 'up_loss','dloss_1']
    RunningAverage(alpha=0.98, output_transform=lambda x: x['tgt_loss']).attach(trainer, 'tgt_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['src_loss']).attach(trainer, 'src_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['vgg_loss']).attach(trainer, 'vgg_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['GP']).attach(trainer, 'GP')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['d_loss']).attach(trainer, 'd_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['down_loss']).attach(trainer, 'down_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['up_loss']).attach(trainer, 'up_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['dloss_1']).attach(trainer, 'dloss_1')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={
                                  'feature_extractor': feature_extractor,
                                  'SR': SR,
                                  'ADAM': optimizer,
                                  'domain_D': domain_classifier,
                                  'res_D': resolution_classifier,
                                  'sr_D': sr_classif_critic
                              })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            fname = os.path.join(args.output_dir, LOGS_FNAME)
            columns = engine.state.metrics.keys()
            values = [str(round(value, 5)) for value in engine.state.metrics.values()]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            i = (engine.state.iteration % len(loader))
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=args.epochs,
                                                                  i=i,
                                                                  max_i=len(loader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            pbar.log_message(message)

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_real_example(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            if (engine.state.iteration - 1) % PRINT_FREQ == 0:
                if not os.path.exists(args.output_dir + '/imgs/domain_adaptation_training/'):
                    os.makedirs(args.output_dir + '/imgs/domain_adaptation_training/')
            px, py, px2, py2, px_up, _, px2_up, _ = engine.state.batch
            img = SR(feature_extractor(px2.cuda()))
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                predtgt_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(img, path)
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                targetY_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(py2, path)
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                targetX_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(px2, path)
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                sourceX_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(px, path)
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                sourceY_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(py, path)
            img = SR(feature_extractor(px.cuda()))
            path = os.path.join(args.output_dir + '/imgs/domain_adaptation_training/',
                                predsrc_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(img, path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            checkpoint_handler(engine, {
                'feature_extractor_{}'.format(engine.state.iteration): feature_extractor,
                'SR_{}'.format(engine.state.iteration): SR,
                'ADAM_{}'.format(engine.state.iteration): optimizer,
                'DOMAIN_D_{}'.format(engine.state.iteration): domain_classifier,
                'RES_D_{}'.format(engine.state.iteration): resolution_classifier,
                'SR_D_{}'.format(engine.state.iteration): sr_classif_critic
            })

        else:
            raise e

    @trainer.on(Events.STARTED)
    def loaded(engine):
        if args.epoch_c != 0:
            engine.state.epoch = args.epoch_c
            engine.state.iteration = args.epoch_c * len(loader)
