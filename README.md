# FingerprintMatching
Project to match different fingerprints from the same person.

# Environment
~~Python 3.6.9 with requirements.txt~~
~~OR, if your hardware does not support it:~~
Python 3.10.0 with requirements_python_3_10_0.txt

# Getting Dataset

tbd

# Replicating Experiments

filepaths will need to be changed in the .sh scripts

Do not slash \\ at the end for folder inputs to bash scripts

## Pretraining
    cd synthetic_data_transfer_learning
    CUDA_VISIBLE_DEVICES=x python3 runner.py \
        --model_path "[desired output folder]/model_weights/embedding_net_weights_printsgan.pth" \
        --data_path "/data/therealgabeguo/printsgan" \
        --output_folder "[desired output folder]/pretrain_results"

## Base Model

### With Pretraining

tbd

### Without Pretraining

## Finger-by-Finger Correlation
    cd dl_models
    bash run_finger_by_finger_balanced.sh [desired output folder (no slash at end)] [cuda num]

## Feature Correlations
    cd dl_models
    bash run_feature_correlation.sh [desired output folder] [cuda num]

## Demographics (Generalizability)

### Race

    cd dl_models
    bash run_race.sh [desired_output_folder] [cuda_num]

### Gender (Generalizability)

    cd dl_models
    bash run_gender.sh [desired_output_folder] [cuda_num]