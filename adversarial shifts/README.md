Code for L2 adversarial shift experiments. Forked form the code provided by Cohen et al. (2019). For CIFAR-10, we suggest that you first download the pre-trained models provided by Cohen et al. [https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing].
For SVHN, you can train the models yourself like this (repeat for noise standard deviation 0.00, 0.12, 0.25, 0.50, 1.00):
```
code/train.py svhn cifar_resnet20  models/svhn/resnet20/noise_[noise standard deviation] --batch 400 --noise [noise standard deviation] --gpu 0
```

 Once trained models are obtained, use the following new scripts (replace cifar10 with svhn as appropriate):

```
code/test_base.py cifar10 [path to directory containing model]  --noise_sd [noise standard deviation]
```

 This will evauate the accuracy of the model under Gaussian noise, and print the number of correctly-classified samples (this information will also be saved in the model directory). Be sure to match the noise specified with the noise that the model was trained under (i.e., use 0.25 for the model trained with sigma = 0.25 noise)

```
code/test_zero_attack.py cifar10 [path to directory containing baseline (sigma = 0) model]
```

This will attack the baseline model using a Carlini-Wagner L2 attack (from IBM ART) and record the attack magnitudes for each sample, as well as whether or not each attack is successful, in the directory containing the classifier.

```
plot_results_CIFAR10.py
```

This will compute certificates and plot the results, along with the undefended attack results. Note that the smoothed model accuracies are currently hard-coded from previous runs of code/test_base.py; however, to plot empirical attack results, code/test_zero_attack.py must be run first; data will then be read from the output of that command.


***

For the attack on distributionally smoothed classifiers, we adapt code from Salman et al. 2019, in the code_smooth_adv directory. Use the following scripts:

```
code_smooth_adv/test_base_attack.py cifar10 [checkpoint directory] [noise standard deviation] [output directory] --skip 10  --batch 400 --N 100  --attack PGD --epsilon [attack epsilon * 255] --num-steps 20 --num-noise-vec 128 
```
 This will attack the smoothed classifier. Be sure to match the noise specified with the noise that the model was trained under (i.e., use 0.25 for the model trained with sigma = 0.25 noise). The attack epsilon is scaled by a factor of 255 (i.e, use 255.0 instead of 1.0).  Use the option "--alt-loss" to use the alternative loss function introduced in the paper, rather than the Salman et al. loss function. Note that only the PGD attack will work correctly.

```
plot_results_smoothed_attacks_CIFAR10.py 
```

Will plot result, with  smoothed attack results added. test_base_attack.py must first be run for a full range of epsilon values (255/8 through 512 in increments of 255/8), and must also be run for "clean" samples (code_smooth_adv/test_base_attack.py cifar10 [checkpoint directory] [noise standard deviation] [output directory] --skip 10  --batch 400 --N 100).

***

To plot results using the "probabilistic" formulation of the distributional attacker Adv' (Figure 10 in the appendix), for which we do not need to estimate the population Wasserstein distance of the attack but rather can compute it by construction, use the command

```
plot_results_probabilistic_attacker_CIFAR10.py
```

Note that this uses the same raw data from the C&W L2 attack as the standard 'plot_results' script; as with the standard script, code/test_zero_attack.py must be run first, and data will then be read from the output of that command.




