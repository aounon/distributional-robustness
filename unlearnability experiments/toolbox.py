import numpy as np
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725, norm = 'inf'):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.norm = norm
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, gaussian_smoothing = False, smoothing_std = 0.0):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        if (gaussian_smoothing):
            gauss_noise = torch.randn((1,8,) + perturb_img.shape[1:],device=perturb_img.device) *smoothing_std
            fake_ones = torch.ones((1,8,) + labels.shape[1:],device=labels.device,dtype=labels.dtype)
            labels = (labels.unsqueeze(1) * fake_ones).reshape((-1,) + labels.shape[1:])
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                if (gaussian_smoothing):
                    logits = model((perturb_img.unsqueeze(1) + gauss_noise).reshape((-1,) + perturb_img.shape[1:]))
                else:
                    logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            if (self.norm == 'inf'):
                eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            elif (self.norm == '2'):
                eta = self.step_size * perturb_img.grad.data/perturb_img.grad.data.reshape(perturb_img.shape[0],-1).norm(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  * (-1)
                eta = eta.nan_to_num()
            else:
                assert False
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            if (self.norm == 'inf'):
                eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            elif (self.norm == '2'):
                eta = (perturb_img.data - images.data)
                eta = (eta /eta.reshape(perturb_img.shape[0],-1).norm(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))*  eta.reshape(perturb_img.shape[0],-1).norm(dim=1).clamp(max=self.epsilon).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                eta = eta.nan_to_num()
            else:
                assert False
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            if (self.norm == 'inf'):
                eta = self.step_size * perturb_img.grad.data.sign() 
            elif (self.norm == '2'):
                eta = self.step_size * perturb_img.grad.data/perturb_img.grad.data.reshape(perturb_img.shape[0],-1).norm(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                assert False
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            if (self.norm == 'inf'):
                eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            elif (self.norm == '2'):
                eta = (perturb_img.data - images.data)
                eta = (eta /eta.reshape(perturb_img.shape[0],-1).norm(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))*  eta.reshape(perturb_img.shape[0],-1).norm(dim=1).clamp(max=self.epsilon).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                assert False
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
