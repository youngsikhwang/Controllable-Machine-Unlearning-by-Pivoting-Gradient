echo "CUP_FID"

cup_path=save_path

for gamma in $(seq 0.1 0.1 0.1)
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
