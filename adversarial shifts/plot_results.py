import re
import scipy.special
import math
from scipy.stats import norm, binom_test
import numpy as np
import torch
from scipy.stats import sem
from statsmodels.stats.proportion import proportion_confint
num_val = 10000
conf = .001
def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
import matplotlib.pyplot as plt

raw_acc_counts = {
	0.12 : 8183,
	0.25 : 7467,
	0.5 : 6421,
	1. : 4829,
}
plt.figure(figsize=(8,4))
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
radii_file = torch.load('models/cifar10/resnet110/noise_0.00/adversarial_accuracy.pth.tar')
radii = radii_file['radii']
max_radii = radii.max()
radii[~radii_file['sucesses']] = max_radii + 0.02
for gamma in np.arange(0,max_radii + 0.01,0.01):
	empirical.append(1.*(radii > gamma).sum()/num_val)
	empirical_x.append((radii*(radii <= gamma)).mean())

empirical.append(empirical[-1])
empirical_x.append(1000.) # Any number outside displayed range should work
print(empirical)
print(empirical_x)
plt.plot(empirical_x, empirical, label ='Undefended', linestyle = '--', color='black')
plt.title('ℓ₂ Adversarial Distribution Shift')
plt.ylabel("Accuracy")
plt.xlabel('Wasserstein Bound (epsilon)')#,fontsize=12)
plt.ylim((0, 1.))
plt.xlim((0, 1.0))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.savefig('L_2_adversarial.png', dpi = 300)
