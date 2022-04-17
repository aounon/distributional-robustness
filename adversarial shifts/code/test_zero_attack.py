import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
import torch.nn as nn
from train_utils import AverageMeter, accuracy, init_logfile, log
from torch.linalg import norm

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method 
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('dir', type=str, help='folder containing model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model_file = torch.load(args.dir + '/checkpoint.pth.tar')
    model = get_architecture(model_file['arch'], args.dataset)
    model.load_state_dict(model_file['state_dict'])

    test_acc,l2s,sucesses = test(test_loader, model)
    print(test_acc)
    print(test_acc * len(test_dataset)/100.)
    torch.save({'correct' : int(round(test_acc * len(test_dataset)/100.)), 'num_samples' : len(test_dataset), 'radii': l2s, 'sucesses': sucesses},args.dir + '/adversarial_accuracy.pth.tar')
def test(loader: DataLoader, model: torch.nn.Module):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    model=model.cuda()
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        input_shape=(3,32,32),
        nb_classes=10,
    )
    l2s = []
    successes = []
    attack = CarliniL2Method(classifier=classifier,batch_size=args.batch)
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # inputs = inputs.cuda()
        # targets = targets.cuda()


        x_test_adv = attack.generate(x=inputs, y = targets)
        x_test_adv = torch.tensor(x_test_adv)

        # compute output
        outputs = model(x_test_adv.cuda())
        success = torch.argmax(outputs,dim=1) != targets.cuda()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets.cuda(), topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print(norm((x_test_adv-inputs).reshape(x_test_adv.shape[0], -1), dim = 1))
        print(success)
        l2s.append( norm((x_test_adv-inputs).reshape(x_test_adv.shape[0], -1), dim = 1) )
        successes.append(success)
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time,  top1=top1, top5=top5))
    return top1.avg, torch.cat(l2s), torch.cat(successes)


if __name__ == "__main__":
    main()
