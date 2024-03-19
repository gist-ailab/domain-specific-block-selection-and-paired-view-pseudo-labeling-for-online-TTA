# Domain Specific Block Selection and Paired-View Pseudo-Labeling for Online Test-Time Adaptation
This is an source code for reproducing the results in our paper "Domain Specific Block Selection and Paired-View Pseudo-Labeling for Online Test-Time Adaptation".


## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda create -n tta python=3.8
conda activate tta 
pip install -r requirements.txt
```

## Classification

<details open>
<summary>Features</summary>
This repository allows to reproduce the results in the paper. A quick overview is given below:

- **Datasets**
  - Dataset will be automatically downloaded. If you have problem, you can manually download the corruption dataset in following links and place it in "classification/data/" folder.
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)

- **Models**
  - For the CIFAR10-C and CIFAR100-C benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.
  
- **Settings**
  - `reset_each_shift` Reset the model state after the adaptation to a domain.
  - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
  - `gradual` Train the model on a sequence of gradually increasing/decreasing domain shifts without knowing when a domain shift occurs.
  - `mixed_domains` Train the model on one long test sequence where consecutive test samples are likely to originate from different domains.

- **Methods**
  - The repository currently supports the following methods: BN-0 (source), BN-1, [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
  [MEMO](https://openreview.net/pdf?id=vn74m_tWu8O), [ETA](https://arxiv.org/abs/2204.02610), [EATA](https://arxiv.org/abs/2204.02610),
  [CoTTA](https://arxiv.org/abs/2203.13591), [AdaContrast](https://arxiv.org/abs/2204.10377), [LAME](https://arxiv.org/abs/2201.05718), 
  [SAR](https://arxiv.org/pdf/2302.12400.pdf), [RoTTA](https://arxiv.org/pdf/2303.13899.pdf),
  [GTTA](https://arxiv.org/abs/2208.07736), [RMT](https://arxiv.org/abs/2211.13081), and [ROID](https://arxiv.org/abs/2306.00650) .


</details>

### Run Experiments
To run the code for CIFAR benchmarks, run the following command first.
```bash
cd classification
```

Then, this code provides config files for all experiments and methods. Simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[cifar10_c/cifar100_c]/[source/norm_test/tent/memo/eata/cotta/adacontrast/lame/sar/rotta/rmt/roid/dplot].yaml
```

Also, to reproduce the results of DPLOT. Run the following Python file with DPLOT config.
```bash
python test_time.py --cfg cfgs/[cifar10_c/cifar100_c]/dplot.yaml
```


### Changing Configurations
Using default setting, the continual setting benchmarks would be conducted. To run DPLOT on CIFAR10-to-CIFAR10-C in the `mixed_domains` setting, the arguments below have to be passed. 
```bash
python test_time.py --cfg cfgs/cifar10_c/dplot.yaml SETTING mixed_domains
```

Also, to run DPLOT on CIFAR10-to-CIFAR10-C in the `gradual` setting, the arguments below have to be passed. 
```bash
python test_time.py --cfg cfgs/cifar10_c/dplot.yaml SETTING gradual
```

### Changing Architectures
Using checkpoints given by RobustBench, we can simply run the adaptation method with various architectures. For example, to run DPLOT on CIFAR10-to-CIFAR10-C using the ResNet18A, the configuration have to be changed. 
```bash
python test_time.py --cfg cfgs/cifar10_c/dplot.yaml SETTING gradual MODEL.ARCH Kireev2021Effectiveness_RLATAugMix
```

The name of the MODEL.ARCH is as follows.

For CIFAR10-C,
- WRN28-10: standard
- WRN40-2A: Hendrycks2020AugMix_WRN
- ResNet18A: Kireev2021Effectiveness_RLATAugMix

For CIFAR100-C,
- ResNext-29A: Hendrycks2020AugMix_ResNeXt
- WRN40-2A: Hendrycks2020AugMix_WRN


### Acknowledgements
This work was partially supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty Aware Agents Learning by Asking Questions) and by LG Electronics and was collaboratively conducted with the Advanced Robotics Laboratory within the CTO Division of the company.

This code is based on the official repository for the following works:

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)
+ SAR [official](https://github.com/mr-eggplant/SAR)
+ RMT [official](https://github.com/mariodoebler/test-time-adaptation)

