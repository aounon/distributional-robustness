Code for experiments on natural transformations for submission titled: Certifying Model Accuracy under Distribution Shift
Adapted from Cohen et. al.'s 'Certified Adversarial Robustness via Randomized Smoothing' code.

train.py: To train models under random transformations
test.py: To obtained certified accuracy under original distribution
tools.py: Contains implementations of various transformations

Example for training a model:

python train.py cifar10 cifar_resnet110 color_shift [path-to-save-model] --batch 400 --noise 0.2 --gpu [gpu index]

Example for testing a model:

python test.py cifar10 [path-to-trained-model] color_shift 2.0 [filename-to-save-accuracy]