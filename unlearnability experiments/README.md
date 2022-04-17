Code for unlearnability experiments. Forked from [code](https://github.com/HanxunH/Unlearnable-Examples) for Huang et al. 2021. **Note than smoothing, L2, and "offline" functionality is currently only configured to work correctly for CIFAR-10, min-min attack, sample-wise attack experiments.**

To reproduce our experiments, run the scripts:
 ```
L_2_make_perturbations_offline.sh
L_2_evaluate_offline.sh
 ```
 
```
L_2_make_perturbations_offline_adaptive.sh
L_2_evaluate_offline_adaptive.sh
```

Finally, to plot results, run:

```
parse_results.py
parse_results_adaptive.py
parse_results_final_main.py
```
