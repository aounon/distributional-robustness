import re
import scipy.special
import math
from scipy.stats import norm, binom_test
import numpy as np
import torch
import pandas
from scipy.stats import sem
from statsmodels.stats.proportion import proportion_confint
num_val = 10000
conf = .001
efficiencies = list([x*255./8. for x in range(1,17)])
threshhold_step = 0.01


def get_empirical_accs(s):
	inits = pandas.read_csv("cifar10_smooth_out_dir/noise_"+str(s) + "/predictions", delimiter='\t')
	inits_a = inits['predict_a'].to_numpy()[None]
	outs_a = []
	outs_b = []
	for j in efficiencies:
		col = pandas.read_csv("cifar10_smooth_out_dir/PGD_"+str(j)+"_noise_"+str(s) + "_mtest_128/predictions", delimiter='\t')
		outs_a.append(col['predict_a'].to_numpy())
		outs_b.append(col['predict_b'].to_numpy())
	outs_a= np.stack(outs_a)
	outs_a_prep = np.concatenate([inits_a,outs_a])
	outs_b= np.stack(outs_b)
	outs_a_rel = outs_a-inits_a
	outs_b_prep = np.concatenate([inits['predict_b'].to_numpy()[None],outs_b])
	#assert ((outs_a_rel > 0).sum() == 0)
	outs_a_rel_eff = outs_a_rel/ np.array(efficiencies)[:,None]
	out_cost = []
	out_accs = []
	for thresh in np.arange(0,-(outs_a_rel_eff.min()) + threshhold_step,threshhold_step):
		last_thresh_eff_a = 15 - np.flip(-outs_a_rel_eff > thresh,axis=0).argmax(axis=0)
		last_thresh_eff_a[np.all(-outs_a_rel_eff <= thresh,axis=0)] = -1
		last_thresh_eff_a = last_thresh_eff_a + 1

		cost = (last_thresh_eff_a ).mean() * 0.125
		acc = outs_b_prep[last_thresh_eff_a,np.arange(1000)].mean()/100
		out_cost.append(cost +  min(2.0,1./thresh) * math.sqrt(math.log(1/0.01)/(2*num_val)))
		out_accs.append(acc)
	return out_cost,out_accs
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
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
eps_range = np.arange(0,1.01,0.01)
for i,noise in enumerate([0.12,0.25,0.5, 1.]):
	certs = []
	for eps in eps_range:
		certs.append(1.*(_lower_confidence_bound(raw_acc_counts[noise], num_val, conf)- scipy.special.erf(eps/(2*math.sqrt(2)*noise))  ))
	print(certs)

	plt.plot(eps_range, certs, label ='Certificate (Noise = ' + str(noise) + ')', linestyle = '-',color = colors[i])
	x,y = get_empirical_accs(noise)
	plt.plot(x,y, label ='Empirical (Noise = ' + str(noise) + ')', linestyle = '--',color = colors[i])

empirical = []
empirical_x = []
radii_file = torch.load('models/cifar10/resnet110/noise_0.00/adversarial_accuracy.pth.tar')
radii = radii_file['radii']
max_radii = radii.max()
radii[~radii_file['sucesses']] = max_radii + 0.02
for gamma in np.arange(0,max_radii + 0.01,0.01):
	empirical.append(1.*(radii > gamma).sum()/num_val)
	empirical_x.append((radii*(radii <= gamma)).mean() + gamma * math.sqrt(math.log(1/0.01)/(2*num_val))) 

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

plt.savefig('plot.png', dpi = 300)
