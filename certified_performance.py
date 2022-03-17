import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture

from tools import *

from statsmodels.stats.proportion import proportion_confint

import numpy as np
from scipy.special import erf

from math import sqrt, pi

parser = argparse.ArgumentParser(description='Evaluating undefended model udner shifted data distribution')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('transform', type=str, choices=TRANSFORMS)
parser.add_argument("smoothing_noise", type=float, help="Noise level used for smoothing")
parser.add_argument('outfile', type=str, help='File to save the model accuracies')
parser.add_argument("--dist_range", type=float, default=10.0, help="Range for Wasserstein distance")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--num_samples", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


def main():
    sigma = args.smoothing_noise
    if args.transform == 'color_shift':
        # sigma = (sqrt(pi) * args.radius) / (2 * sqrt(2))
        custom_transform = torchvision.transforms.Lambda(lambda x: color_shift(x, sigma=sigma))
    elif args.transform == 'hue_shift':
        custom_transform = torchvision.transforms.Lambda(lambda x: hue_shift(x, rand_max=sigma))
    elif args.transform == 'sv_shift':
        # sigma = args.radius * 2
        custom_transform = torchvision.transforms.Lambda(lambda x: sv_shift(x, rand_max=sigma))
    else:
        print('Unrecognized transformation... exiting.')
        quit()

    # Loading data
    dataset = get_dataset(args.dataset, 'test', custom_transform)
    pin_memory = (args.dataset == "imagenet")

    sampler = RandomSampler(dataset, replacement=True, num_samples=args.num_samples)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    # Loading trained model
    checkpoint = torch.load(args.base_classifier)
    model = get_architecture(checkpoint["arch"], args.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    # Switching to eval mode
    model.eval()

    num_correct = 0

    flag = False #True
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if flag:
                print('Saving input images to file.')
                imgsave(torchvision.utils.make_grid(inputs[:64]), 'input_grid.png')
                flag = False

            # print(i)
            inputs = inputs.cuda()
            targets = targets.cuda()

            # compute output
            outputs = model(inputs)
            output_class = torch.argmax(outputs, dim=1)

            # counting correct predictions
            correct = torch.where(output_class == targets, 1, 0)
            num_correct += torch.sum(correct)

    num_correct = num_correct.item()

    cert_acc = proportion_confint(num_correct, args.num_samples, alpha=2 * args.alpha, method="beta")[0]

    print('Certified accuracy = %.3f' % cert_acc)

    eps = np.arange(0, args.dist_range, args.dist_range/100)
    eps = np.append(eps, args.dist_range)

    if args.transform == 'color_shift':
        cert_acc_eps = cert_acc - erf(eps / (2 * np.sqrt(2) * sigma))
    elif args.transform == 'sv_shift':
        cert_acc_eps = cert_acc - eps / sigma
    else:
        print('Unrecognized transformation... exiting.')
        quit()

    cert_acc_eps = np.maximum(cert_acc_eps, 0)

    f = open(args.outfile, 'w+')
    for i in range(eps.shape[0]):
        f.write('%.1f %.3f\n' % (eps[i], cert_acc_eps[i]))
    f.close()


if __name__ == "__main__":
    main()
