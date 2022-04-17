import re
import scipy.special
import math
from scipy.stats import norm, binom_test
import numpy as np
from scipy.stats import sem
from statsmodels.stats.proportion import proportion_confint
num_val = 10000
conf = .001
def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
import matplotlib.pyplot as plt
eps_range =[0.2, 0.4, 0.6, 0.8, 1.0]
for sigma in [0.0, 0.1, 0.2, 0.4]:
	clean_val_accs = []
	poison_val_accs = []
	test_accs = []
	lower_bounds = []
	clean_val_accs_std = []
	poison_val_accs_std = []
	test_accs_std = []
	lower_bounds_std = []
	for eps in eps_range:
		clean_val_accs_sub = []
		poison_val_accs_sub = []
		test_accs_sub = []
		lower_bounds_sub = []
		for replicates in ['/0/','/1/','/2/','/3/','/4/']:
			f =  open('experiments/cifar10/offline_min-min_samplewise/CIFAR10-eps='+str(eps)+'-se=0.01-base_version=resnet18-norm-2'+replicates+ 'sigma_'+str(sigma)+'/resnet18/resnet18.log', 'r')
			str_file = f.read()
			f.close()
			clean_val_accs_sub.append(float(re.findall('clean_val[^\n]+Eval acc: (.*)\n',str_file)[-1]))
			poison_val_accs_sub.append(float(re.findall('poison_val[^\n]+Eval acc: (.*)\n',str_file)[-1]))
			test_accs_sub.append(float(re.findall('test[^\n]+Eval acc: (.*)\n',str_file)[-1]))
			if (sigma != 0):
				lower_bounds_sub.append(100.*(_lower_confidence_bound(int(num_val* poison_val_accs_sub[-1]/100.), num_val, conf)- scipy.special.erf(eps/(2*math.sqrt(2)*sigma))  ))
		clean_val_accs.append(np.array(clean_val_accs_sub).mean()/100.)
		clean_val_accs_std.append(sem(np.array(clean_val_accs_sub))/100.)
		poison_val_accs.append(np.array(poison_val_accs_sub).mean()/100.)
		poison_val_accs_std.append(sem(np.array(poison_val_accs_sub))/100.)
		test_accs.append(np.array(test_accs_sub).mean()/100.)
		test_accs_std.append(sem(np.array(test_accs_sub))/100.)
		if (sigma != 0):
			lower_bounds.append(np.array(lower_bounds_sub).mean()/100.)
			lower_bounds_std.append(sem(np.array(lower_bounds_sub))/100.)
	plt.figure()
	plt.style.use('bmh')
	plt.errorbar(eps_range, poison_val_accs, yerr= poison_val_accs_std,label ='Unlearnable Validation Accuracy', color = '#348ABD', linestyle = ':')
	plt.errorbar(eps_range, clean_val_accs,yerr=clean_val_accs_std, label ='Clean Validation Accuracy', color='r', linestyle = '--')
	plt.errorbar(eps_range, test_accs, yerr = test_accs_std, label ='Clean Test Accuracy', color = 'g', linestyle = '--')
	if (sigma != 0):
		plt.errorbar(eps_range, lower_bounds,yerr=lower_bounds_std, label ='Certified Accuracy', color = '#348ABD', linestyle = '-')
	if (sigma == 0):
		plt.title('Undefended')
	else:
		plt.title('Non-Adaptive Attack, σ = ' + str(sigma))
	plt.ylabel("Accuracy")
	plt.xlabel('Wasserstein Bound (epsilon)')
	plt.ylim((0, 1.))
	plt.xlim((0, 1.0))
	plt.tight_layout()
	plt.legend()
	plt.savefig('offline_nonadaptive_orig_sigma_'+str(sigma)+'.png', dpi = 300)
