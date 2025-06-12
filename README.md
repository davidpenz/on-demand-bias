## Source code repository for RecSys 2025 short paper: 
# Mitigating Latent User Biases in Pre-trained VAE Recommendation Models via On-demand Input Space Transformation

## Install

Get Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Execute the dependency installation
```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -n bias --file environment.yaml 
conda activate bias
```
## Data Preprocessing & Experiment Setup
Setup the variables `ROOT_DIR` and `OUTPUT_DIR` in `preprocess_dataset.py`. `ROOT_DIR` represents the path to the raw dataset directory. `OUTPUT_DIR` specifies the path where the preprocessed dataset will be stored.

Setup the variables `_base_local_dataset_path_map` and `_base_local_results_path_map` in `src/config/data_paths.py`. `_base_local_dataset_path_map` should be equivalent to the entry in `OUTPUT_DIR`. `_base_local_results_path_map` specifies the directory where experiment results will be stored.

```bash
conda activate bias
python preprocess_dataset.py  
```

The paper presents experiments for three datasets and four different model variations. The experiment configurations are stored in `config_files/` and structured in subfolders following the format `model/dataset.yaml`. 

For the `MultVAE_transform` and `MultVAE_transform+adv` models, the corresponding configuration file includes the path to the pre-trained model. This is by default set to the pre-trained models located at `pretrained/`. To train ```MultVAE_transform``` and ```MultVAE_transform+adv```, a pre-trained ```MultVAE``` must first be trained and then copied to the folder structure ```pretrained/DATASET/FOLD/train/best_model_utility.pt```, e.g., ```pretrained/ekstra/0/train/best_model_utility.pt```.

The model architecture used for all experiments can be found at [src/recsys_models/mask_mult_vae.py (MaskMultVAE)](src/recsys_models/mask_mult_vae.py).

_**Remark:** The current implementation is optimized for GPUs only._

## Running
Below are sample commands to train and evaluate the MultVAE model for the movielens-1M dataset. 

To train a model:
```bash
conda activate bias
python run.py --config config_files/MultVAE/ml1m.yaml --n_parallel 1 --gpus 0 --n_folds 5
```
To perform inference attack of a pretrained model:
```bash
conda activate bias
python run_atk.py --experiment /results/path/ml-1m/MaskMultVAE--YYYY-MM-DD_hh-mm-ss/ \ 
    --atk_config config_files/ml1m_gender_atk.yaml --n_parallel 1 --gpus 0 --n_folds 5 \
    --model_pattern "**/*train*/**/*model_epoch*.pt"
```
