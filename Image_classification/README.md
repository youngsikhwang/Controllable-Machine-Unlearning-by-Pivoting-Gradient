# Image Classification Task

This is the implementation of CUP for image classification task. The code structure of this work is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse), [Unlearn Saliency](https://github.com/OPTML-Group/Unlearn-Saliency.git) codebase.

## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. First, train the original model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

    For example, we can train the ResNet-18 on CIFAR-10 by run this code:
    ```bash
    python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 200
    ```

2. Unlearn
    
    We conduct the experiments on 5 different random seeds and 20 hyperparameter settings. For example, run the below code to unlearn class 0 with CUP on CIFAR-10 dataset.

    ```bash
    echo "Run CUP experiment"

    original_path='original model path'
    save_dir='dir to save the model after unlearning'

    lr_list=(0.001 0.0001)

    for seed in $(seq 1 1 5)
    do
        for lr in "${lr_list[@]}"
        do
            for gamma in $(seq 0.01 0.01 0.99)
            do
                echo "Running experiment with seed=$seed and unlearning intensity=$gamma"
                python /main_forget.py \
                    --unlearn GA_cup \
                    --dataset cifar10 \
                    --gamma $gamma \
                    --num_classes 10 \
                    --seed $seed \
                    --unlearn_epochs 5 \
                    --unlearn_lr $lr \
                    --class_to_replace 0 \
                    --num_indexes_to_replace 4500 \
                    --model_path $original_path \
                    --save_dir $save_dir
            done
        done
    done

    echo "End experiment"

    ```
3. Evaluation

    After unlearning, the results will be recorded in 
    ```./result/dataset method_dataset_seed_hyperparameter_lr.txt``` files. For calulate hypervolume indicator, we use pygmo package. Please refer ```visualization_cifar10.ipynb``` file.