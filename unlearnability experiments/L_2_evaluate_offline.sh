export config_path=configs/cifar10
export dataset_type=CIFAR10
export poison_dataset_type=PoisonCIFAR10
export attack_type=min-min
export perturb_type=samplewise
export base_version=resnet18
export num_steps=20
export universal_stop_error=0.01
export universal_train_target='train_dataset'

export model_name=$base_version
export poison_rate=1.0
export train_portion=0.8

for replicate_loop in 0 1 2 3 4
do
for epsilon_loop in  0.2 0.4 0.6 0.8 1.0
do
	export epsilon=$epsilon_loop
	export step_size=`echo "scale=2;$epsilon_loop/10.0" | bc`
	export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}-norm-2
	export exp_path=experiments/cifar10/offline_${attack_type}_${perturb_type}/${exp_args}/${replicate_loop}
	for sigma in 0.0 0.1 0.2 0.4
	do
	 export exp_name=${exp_path}/sigma_${sigma}

		# Poison Training
		python3 -u main_offline.py --train_portion $train_portion --version  $model_name --gaussian_noise $sigma --exp_name $exp_name --config_path $config_path --train_data_type $poison_dataset_type  --poison_rate $poison_rate  --perturb_type $perturb_type --perturb_tensor_filepath_train ${exp_path}/train_perturbation.pt --perturb_tensor_filepath_valid ${exp_path}/valid_perturbation.pt --perturb_tensor_filepath_proxy ${exp_path}/proxy_perturbation.pt  --train
	done

done
done