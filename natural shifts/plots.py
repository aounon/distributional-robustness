import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import csv
from scipy.special import erf
import numpy as np

parser = argparse.ArgumentParser(description='Plots')
parser.add_argument('accuracy_file', type=str, help='CSV file containing certified accuracies')
parser.add_argument('transform', type=str)
parser.add_argument('outfile', type=str, help='PNG file to save the plots')
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--title', type=str, help='Plot title', default='Color Shift')
args = parser.parse_args()


def main():
    plt.rcParams.update({'font.size': 12})
    mpl.style.use('bmh')
    with open(args.accuracy_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            sigma = float(row[0])
            acc = float(row[1])

            step = args.radius / 100
            eps = np.arange(0, args.radius, step)
            eps = np.append(eps, args.radius)
            # print(eps)

            if args.transform == 'color_shift':
                cert_acc = acc - erf(eps / (2 * np.sqrt(2) * sigma))
            elif args.transform == 'sv_shift':
                cert_acc = acc - (eps/sigma)
            else:
                print('Unrecognized transformation... exiting.')
                quit()

            cert_acc = np.maximum(cert_acc, 0)
            label = 'Smoothing Noise = %.1f' % sigma
            plt.plot(eps, cert_acc, label=label)

    # plt.grid()
    plt.ylim([0.0, 1.0])
    plt.xlabel('Wasserstein Bound (epsilon)')
    plt.ylabel('Certified Accuracy')
    plt.title(args.title)
    plt.legend()
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
