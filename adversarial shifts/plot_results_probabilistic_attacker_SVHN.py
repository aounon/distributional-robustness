import re
import scipy.special
import math
from scipy.stats import norm, binom_test
import numpy as np
import torch
from scipy.stats import sem
from statsmodels.stats.proportion import proportion_confint
num_val = 26032
conf = .001
def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
import matplotlib.pyplot as plt

raw_acc_counts = {
	0.12 : 24156,
	0.25 : 22228,
	0.5 : 17432,
	1. : 11101,
}
plt.figure()
plt.style.use('bmh')
eps_range = np.arange(0,1.01,0.01)
for noise in [0.12,0.25,0.5, 1.]:
	certs = []
	for eps in eps_range:
		certs.append(1.*(_lower_confidence_bound(raw_acc_counts[noise], num_val, conf)- scipy.special.erf(eps/(2*math.sqrt(2)*noise))  ))
	print(certs)
	plt.plot(eps_range, certs, label ='Smoothing Noise = ' + str(noise), linestyle = '-')
empirical = []
empirical_x = []
radii_file = torch.load('models/svhn/resnet20/noise_0.00/adversarial_accuracy.pth.tar')
radii = radii_file['radii'].cuda()
max_radii = radii.max().item()
radii[~radii_file['sucesses']] = max_radii + 0.02

for gamma in np.arange(0,max_radii + 0.01,0.01):
	empirical.append((1.*((radii > gamma).int()*(1. - (radii_file['sucesses']).int()*(  torch.where( (torch.isnan(gamma/radii) |  torch.isinf(gamma/radii) ) , torch.tensor(0).float().cuda(),  gamma/radii ))) ).sum()/num_val).item())
	empirical_x.append(gamma)

empirical.append(empirical[-1])
empirical_x.append(1000.) # Any number outside displayed range should work
print(empirical)
print(empirical_x)
plt.plot(empirical_x, empirical, label ='Undefended', linestyle = '--', color='black')
plt.title('ℓ₂ Adversarial Distribution Shift (Probabilistic Attacker)')
plt.ylabel("Accuracy")
plt.xlabel('Wasserstein Bound (epsilon)')#,fontsize=12)
plt.ylim((0, 1.))
plt.xlim((0, 1.0))
plt.legend()
plt.tight_layout()

plt.savefig('L_2_adversarial_probabilistic_SVHN.png', dpi = 300)
