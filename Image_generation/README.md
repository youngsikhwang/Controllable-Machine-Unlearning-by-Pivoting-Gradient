# Image Generation Task
This is the code implementation of CUP for image generation task. The code structure of this task is adapted from the [DDIM](https://github.com/ermongroup/ddim) and [SA](https://github.com/clear-nus/selective-amnesia/tree/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm) and [Unlearn Saliency](https://github.com/OPTML-Group/Unlearn-Saliency.git) codebase.

## Requirements
Install the requirements using a `conda` environment:
```
conda create --name cup python=3.8
conda activate cup
pip install -r requirements.txt
```

## Unlearning

1. First train a conditional DDPM on all 10 CIFAR10 classes. 

   We demonstrate the code to run CUP on CIFAR10.
   For instance, train the original model by conduct

   ```bash
   python train.py --config cifar10_train.yml --mode train
   ```

2. Unlearning with CUP

   ```bash
   echo "CUP_unlearn"

   original_path=path_to_original_model

   for gamma in $(seq 0.1 0.1 0.9)
   do
      for forget_class in $(seq 0 4)
      do
         python train.py\
         --config cifar10_cup.yml\
         --ckpt_folder $original_path\
         --label_to_forget $forget_class \
         --mode cup\
               --gamma $gamma\
               --alpha 5e-3\
               --method ga
      done
   done
   ```

   This should create another folder in `results/cifar10/unlearn/{method_name}`. 

   You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.

   You can experiment with forgetting different method using the `--method` flag, but we will consider forgetting with gradient ascent (ga) here.


## Evaluation
1. for evaluation,
  ```bash
   echo "CUP_FID"

   cup_path=save_path

   for gamma in $(seq 0.1 0.1 0.9)
   do
      for class in $(seq 0 4)
      do
         python sample.py\
         --config cifar10_sample.yml\
         --ckpt_folder $cup_path/${class}_${gamma}\
         --mode sample_fid\
         --classes_to_generate "x$class" \
         --n_samples_per_class 1000
      done
   done


   for gamma in $(seq 0.1 0.1 0.9)
   do
      for class in $(seq 0 4)
      do
      python evaluator.py ${cup_path}/${class}_${gamma}/fid_samples_guidance_2.0_excluded_class_${class} cifar10_without_label_${class}
      done
   done
   ```
