# Distributional Robustness

Code for experiments on natural transformations for submission titled: [Certifying Model Accuracy under Distribution Shift](https://arxiv.org/abs/2201.12440).
It is adapted from the publicly availabe [code repository](https://github.com/locuslab/smoothing) for the paper [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) by (Cohen et al. 2019).

train.py: To train models under random transformations

test.py: To obtained certified accuracy under original distribution

tools.py: Contains implementations of various transformations

Example for training a model:

```
python train.py cifar10 cifar_resnet110 color_shift [path-to-save-model] --batch 400 --noise 0.2 --gpu [gpu index]
```

Example for testing a model:

```
python test.py cifar10 [path-to-trained-model] color_shift 2.0 [filename-to-save-accuracy]
```
