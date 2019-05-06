import os
import torchvision.utils as vutils
import warnings
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.options import args

PRINT_FREQ = 200
REAL_IMG_FNAME = '{:04d}_{:04d}_hr.png'
FAKE_IMG_FNAME = '{:04d}_{:04d}_pred.png'
TRAIN_IMG_FNAME = '{:04d}_{:04d}_lr.png'
DOWN_IMG_FNAME = '{:04d}_{:04d}_downsampler.png'
LOGS_FNAME = 'logs.tsv'
PLOT_FNAME = 'plot.svg'
CKPT_PREFIX = 'networks'


def start(trainer, SR, D, vgg, loader, schedulerD, schedulerG, optimizerD, optimizerG,resume_epoch,resume_iter):

    timer = Timer(average=True)

    checkpoint_handler = ModelCheckpoint(args.output_dir, 'training', save_interval=1, n_saved=10, require_empty=False)

    monitoring_metrics = ['dloss_real', 'dloss_fake', 'd_loss', 'GP', 'WD', 'VGG', 'gloss']
    RunningAverage(alpha=0.98, output_transform=lambda x: x['dloss_real']).attach(trainer, 'dloss_real')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['dloss_fake']).attach(trainer, 'dloss_fake')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['GP']).attach(trainer, 'GP')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['d_loss']).attach(trainer, 'd_loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['WD']).attach(trainer, 'WD')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['VGG']).attach(trainer, 'VGG')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['gloss']).attach(trainer, 'gloss')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={
                                  'SR': SR,
                                  'D': D,
                                  'VGG': vgg
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
            px, y = engine.state.batch
            img = SR(px.cuda())
            path = os.path.join(args.output_dir, FAKE_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(img, path)
            path = os.path.join(args.output_dir, REAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(y, path)
            path = os.path.join(args.output_dir, TRAIN_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            vutils.save_image(px, path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def LRstep(engine):
        schedulerD.step()
        schedulerG.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        try:
            import matplotlib as mpl
            mpl.use('agg')

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.read_csv(os.path.join(args.output_dir, LOGS_FNAME), delimiter='\t')
            x = np.arange(1, engine.state.iteration + 1, PRINT_FREQ)
            _ = df.plot(subplots=True, figsize=(20, 20), grid=True, xticks=x)
            _ = plt.xlabel('Iteration number')
            fig = plt.gcf()
            path = os.path.join(args.output_dir, PLOT_FNAME)

            fig.savefig(path)

        except ImportError:
            warnings.warn('Loss plots will not be generated -- pandas or matplotlib not found')

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            create_plots(engine)
            checkpoint_handler(engine, {
                'netG_{}'.format(engine.state.iteration): SR,
                'netD_{}'.format(engine.state.iteration): D,
                'optim_D_{}'.format(engine.state.iteration): optimizerD,
                'optim_G_{}'.format(engine.state.iteration): optimizerG,
                'sched_D_{}'.format(engine.state.iteration): schedulerD,
                'sched_G_{}'.format(engine.state.iteration): schedulerG
            })

        else:
            raise e

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        engine.state.iteration = resume_iter
        engine.state.epoch = resume_epoch
