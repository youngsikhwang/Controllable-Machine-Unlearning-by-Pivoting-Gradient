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