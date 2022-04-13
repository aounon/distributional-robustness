import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
import os

from tools import *

parser = argparse.ArgumentParser(description='Certified accuracy under hue shift for different training noise levels')
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
    for max_angle in range(0, 190, 20):
        print('Max angle = %d' % max_angle)

        # Random hue shift in the range [-180, 180] degrees
        custom_transform = torchvision.transforms.Lambda(lambda x: hue_shift(x, rand_max=180))

        # Loading data
        dataset = get_dataset('cifar10', 'test', custom_transform)

        sampler = RandomSampler(dataset, replacement=True, num_samples=args.num_samples)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch,
                                 num_workers=args.workers)

        # Creating model path for a given training noise
        model_path = os.path.join(args.path, 'noise_' + str(max_angle), 'checkpoint.pth.tar')
        # print('Model path: ' + model_path)

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
        f.write('%d %.3f\n' % (max_angle, cert_acc))
        f.flush()

    f.close()


if __name__ == "__main__":
    main()
