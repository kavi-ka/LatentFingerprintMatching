# Usage: bash run_base_model_no_pretrain.sh output_folder cuda_num
# Purpose: to train our cross-finger recognition model, without pretraining

SD302='/data/therealgabeguo/fingerprint_data/sd302_split'

SD302_latent='/data/albert/302_latent_data_split'
SD302_latent_toy='/data/albert/latent_302/latent_8bit_toy'

SD301='/data/therealgabeguo/fingerprint_data/sd301_split'
SD300='/data/therealgabeguo/fingerprint_data/sd300a_split'
RIDGEBASE='/data/therealgabeguo/fingerprint_data/RidgeBase_Split'
BASED_WEIGHTS=$1/model_weights/full_based_model.pth
PROVING_CORRELATION_FOLDER="$1/paper_results/proving_correlation"
GENERAL_CORRELATION_FOLDER="$PROVING_CORRELATION_FOLDER/general"

######
# Training base model
######
CUDA_VISIBLE_DEVICES=$2 python3 latent_test.py \
    --datasets "${SD302_latent} ${SD302}" \
    --posttrained-model-path $BASED_WEIGHTS \
    --temp_model_dir '../temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100
