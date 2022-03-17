import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
import os

from tools import *
import numpy as np

parser = argparse.ArgumentParser(description='Certified accuracy under color shift for different training noise levels')
parser.add_argument("path", type=str, help="path to saved pytorch models with different trainning noise levels")
parser.add_argument('outfile', type=str, help='File to save the plots')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--num_samples", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


def main():

    f = open(args.outfile, 'w+')
    for train_sig in np.arange(0.0, 1.01, 0.1):
        print('Training sigma = %.1f' % train_sig)

        # Random hue shift in the range [-180, 180] degrees
        custom_transform = torchvision.transforms.Lambda(lambda x: random_channel(x))

        # Loading data
        dataset = get_dataset('cifar10', 'test', custom_transform)

        sampler = RandomSampler(dataset, replacement=True, num_samples=args.num_samples)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch,
                                 num_workers=args.workers)

        # Creating model path for a given training noise
        if train_sig == 1.0:
            model_path = os.path.join(args.path, 'noise_10/checkpoint.pth.tar')
        else:
            model_path = os.path.join(args.path, 'noise_0' + str(int(train_sig*10)), 'checkpoint.pth.tar')
        print('Model path: ' + model_path)

        # Loading trained model
        checkpoint = torch.load(model_path)
        model = get_architecture(checkpoint["arch"], 'cifar10')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        # Switching to eval mode
        model.eval()

        # Computing certified accuracy
        cert_acc = acc_lbd(model, data_loader, args.num_samples, args.alpha)

        print('Certified accuracy = %.3f' % cert_acc)
        f.write('%.1f %.3f\n' % (train_sig, cert_acc))
        f.flush()

    f.close()


if __name__ == "__main__":
    main()
