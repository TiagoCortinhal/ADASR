import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_workers',
                    type=int, default=10,
                    help='number of data loading workers')

parser.add_argument('--batch-size',
                    type=int, default=100,
                    help='input batch size')

parser.add_argument('--epochs',
                    type=int, default=300,
                    help='number of epochs to train for')

parser.add_argument('--device',
                    default='cuda:0',
                    help='Which device to use')

parser.add_argument('--learning_rate',
                    type=float, default=1e-4,
                    help='Learning Rate')

parser.add_argument('--beta-1',
                    type=float, default=0.,
                    help='beta_1 for adam')

parser.add_argument('--no-cuda',
                    action='store_true',
                    help='disables cuda')

parser.add_argument('--output-dir',
                    default='./out/v2/',
                    help='directory to output images and model checkpoints')

parser.add_argument('--saved_SR',
                    default='',
                    help='path to SR checkpoint')

parser.add_argument('--saved_D',
                    default='',
                    help='path to D checkpoint')

parser.add_argument('--saved_vgg',
                    default='',
                    help='path to VGG checkpoint')

parser.add_argument('--seed',
                    type=int, default=23,
                    help='manual seed')

parser.add_argument('--factor',
                    type=int, default=2,
                    help='Downscalling factor')

parser.add_argument('--eval', action='store_true')

parser.add_argument('--eval-epoch', type=int, default=0)

parser.add_argument('--theta', type=int, default=1)

parser.add_argument('--gamma', type=int, default=10)

parser.add_argument('--train-type', default='dann')

parser.add_argument('--epoch-c', type=int, default=0)

args = parser.parse_args()
