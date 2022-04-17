export config_path=configs/cifar10
export dataset_type=CIFAR10
export poison_dataset_type=PoisonCIFAR10
export attack_type=min-min
export perturb_type=samplewise
export base_version=resnet18
export num_steps=20
export universal_stop_error=0.01
export universal_train_target='train_dataset'
for replicate_loop in 0 1 2 3 4
do
for epsilon_loop in 0.2 0.4 0.6 0.8 1.0
do
	export epsilon=$epsilon_loop
	export step_size=`echo "scale=2;$epsilon_loop/10.0" | bc`
	export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}-norm-2
	export exp_path=experiments/cifar10/offline_${attack_type}_${perturb_type}/${exp_args}/${replicate_loop}
	python3 perturbation_offline.py --config_path $config_path  --exp_name $exp_path  --version $base_version --train_data_type  $dataset_type --noise_shape 20000 3 32 32 --epsilon  $epsilon  --num_steps $num_steps --step_size $step_size --attack_type $attack_type  --perturb_type $perturb_type  --universal_train_target  $universal_train_target --universal_stop_error $universal_stop_error --norm 2

done
done