Code for L2 adversarial shift experiments. Forked form the code provided by Cohen et al. (2019). We suggest that you first download the pre-trained models for CIFAR-10 provided by Cohen et al. [https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing]. One downloaded, use the following new scripts:

```
code/test_base.py cifar10 [path to directory containing model]  --noise_sd [noise standard deviation]
```

 This will evauate the accuracy of the model under Gaussian noise, and print the number of correctly-classified samples (this information will also be saved in the model directory). Be sure to match the noise specified with the noise that the model was trained under (i.e., use 0.25 for the model trained with sigma = 0.25 noise)

```
code/test_zero_attack.py cifar10 [path to directory containing baseline (sigma = 0) model]
```

This will attack the baseline model using a Carlini-Wagner L2 attack (from IBM ART) and record the attack magnitudes for each sample, as well as whether or not each attack is successful, in the directory containing the classifier.

```
plot_results.py
```

This will compute certificates and plot the results, along with the undefended attack results. Note that the smoothed model accuracies are currently hard-coded from previous runs of code/test_base.py; however, to plot empirical attack results, code/test_zero_attack.py must be run first; data will then be read from the output of that command.
