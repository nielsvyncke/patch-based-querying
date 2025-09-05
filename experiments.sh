structures="1"
k="1"
input_dirs="VIB" #"EMBL/ EPFL/ VIB/"
batch_sizes="10 20 30 50 100"
models="ae_pretrained vae_pretrained ae_finetuned vae_finetuned"
set -e

for input_dir in $input_dirs
do
    for model in $models
    do
        for batch_size in $batch_sizes
        do
            for dims in $(seq 80 10 80)
            do
                python search_pipeline.py --batch_size $batch_size --encoder $model --dims $dims  --structure 1 --no_eval --num_neighbors 1\
                        --input_dir $input_dir 
            done
        done
    done
    # python3 PR\ curves.py --dataset $input_dir
done
