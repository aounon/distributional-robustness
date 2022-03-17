# visualize transformed images
import argparse
from datasets import get_dataset, DATASETS
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from math import pi, sqrt
from tools import *

parser = argparse.ArgumentParser(description='visualize transformations')
parser.add_argument("dataset", type=str, choices=DATASETS)
# parser.add_argument('transform', type=str, choices=TRANSFORMS)
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--eps", type=float, default=0.5)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
args = parser.parse_args()


def main():
    sigma = (3 * sqrt(pi) * args.eps) / (2 * sqrt(2))
    print(sigma)

    no_transform = torchvision.transforms.Lambda(lambda x: x)
    CS_transform = torchvision.transforms.Lambda(lambda x: color_shift(x, sigma=sigma))
    # CS_transform = torchvision.transforms.Lambda(lambda x: random_channel(x))
    # HS_transform = torchvision.transforms.Lambda(lambda x: hue_shift(x))
    HS_transform = torchvision.transforms.Lambda(lambda x: random_channel(x))
    # SV_transform = torchvision.transforms.Lambda(lambda x: sv_shift(x, rand_max=2.0))
    SV_transform = torchvision.transforms.Lambda(lambda x: random_channel(x))

    # Loading data
    dataset_original = get_dataset(args.dataset, 'test', no_transform)
    dataset_CS = get_dataset(args.dataset, 'test', CS_transform)
    dataset_HS = get_dataset(args.dataset, 'test', HS_transform)
    dataset_SV = get_dataset(args.dataset, 'test', SV_transform)
    pin_memory = (args.dataset == "imagenet")

    data_loader_original = DataLoader(dataset_original, shuffle=False, batch_size=args.batch,
                                      num_workers=args.workers, pin_memory=pin_memory)
    data_loader_CS = DataLoader(dataset_CS, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
    data_loader_HS = DataLoader(dataset_HS, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
    data_loader_SV = DataLoader(dataset_SV, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)

    _, (images_original, _) = next(enumerate(data_loader_original))
    _, (images_CS, _) = next(enumerate(data_loader_CS))
    _, (images_HS, _) = next(enumerate(data_loader_HS))
    _, (images_SV, _) = next(enumerate(data_loader_SV))

    img_idx = [17, 10, 16, 23, 64, 75, 104, 91, 1, 9, 29, 60, 52, 44, 45]

    images = torch.cat([images_original[img_idx], images_CS[img_idx], images_HS[img_idx], images_SV[img_idx]])

    # start = 0
    # end = start + 64
    # imgsave(torchvision.utils.make_grid(images_original[start:end]), args.outfile)

    imgsave(torchvision.utils.make_grid(images, nrow=len(img_idx)), args.outfile)


if __name__ == "__main__":
    main()
