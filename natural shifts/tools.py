import torch
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from random import randrange

from kornia.color import rgb_to_hsv, hsv_to_rgb
from math import pi

TRANSFORMS = ["color_shift", "color_scale", "hue_shift", "sv_shift", "sv_scale"]


def color_shift(img: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Function to randomly scale the RGB channels of an image relative to each other.
    # img tensor is expected to be of shape [3 x H x W] with values in [0, 1]
    # scaled such that the maximum value is 1.

    exponents = sigma * torch.randn(3)
    exponents = exponents - torch.max(exponents)
    scale = torch.pow(2, exponents)    # sampling a random scaling factor for each color
    # print(scale)
    scale = torch.unsqueeze(torch.unsqueeze(scale, dim=-1), dim=-1)
    img = img * scale
    return img / torch.max(img)


def random_channel(img: torch.Tensor) -> torch.Tensor:
    scale = torch.zeros(3)
    scale[randrange(3)] = 1
    scale = torch.unsqueeze(torch.unsqueeze(scale, dim=-1), dim=-1)
    img = img * scale
    # print(torch.max(img[0]), torch.max(img[1]), torch.max(img[2]))
    return img / torch.max(img)


def hue_shift(img: torch.Tensor, rand_max: float = 180) -> torch.Tensor:
    # Function to randomly shift the hue of an image by an angle (degrees) in [-rand_max, +rand_max]
    # img tensor is expected to be of shape [3 x H x W] with values in [0, 1].
    # rand_max is upper limit (in degrees) on the range of the uniformly sampled angle

    img_hsv = rgb_to_hsv(img)       # converting image to HSV format
    # print(torch.min(img_hsv[0]), torch.max(img_hsv[0]))

    # smapling an angle in the range and converting to radians
    angle = 2 * pi * rand_max * ((2 * torch.rand(1)) - 1) / 360

    img_hsv[0] = img_hsv[0] + angle
    img_hsv[0] = img_hsv[0] - (2 * pi * torch.floor(img_hsv[0]/(2*pi)))
    # print(torch.min(img_hsv[0]), torch.max(img_hsv[0]))
    return hsv_to_rgb(img_hsv)      # TF.adjust_hue(img, np.random.uniform(low=-0.5, high=0.5))


def sv_shift(img: torch.Tensor, rand_max: float = 1.0) -> torch.Tensor:
    # Function to randomly perturb the saturation and brightness of an image.
    # img tensor is expected to be of shape [3 x H x W] with values in [0, 1].
    img_hsv = rgb_to_hsv(img)       # converting image to HSV format
    img_sv = img_hsv[1:]            # separating out saturation and value
    exponents = rand_max * torch.rand(2)  # sampling shift values
    exponents = torch.unsqueeze(torch.unsqueeze(exponents, dim=-1), dim=-1)
    scale = torch.pow(2, exponents) - 1    # sampling a scaling factor
    sv_mean = torch.mean(img_sv, dim=(1, 2), keepdim=True)
    img_sv = img_sv + (scale * sv_mean)
    img_sv = img_sv / torch.max(img_sv)
    img_hsv[1:] = img_sv
    return hsv_to_rgb(img_hsv)


def imgsave(img, filename):
    # Function to save image in a file.
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename) # plt.draw()
    plt.close()


def acc_lbd(model: torch.nn.Module, data_loader, num_samples: int = 100000, alpha: float = 0.001):
    # Function to compute an empirical lower bound on the accuracy of the model

    num_correct = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
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
    # print(num_correct)
    cert_acc = proportion_confint(num_correct, num_samples, alpha=2 * alpha, method="beta")[0]

    return cert_acc

