<!-- SDT-IDS: Spatial Data Transformation For Elevating Intrusion Detection Efficiency in IoT Networks
= -->

- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [Data Preprocessing](#data-preprocessing)
  - [Image Data Generation](#image-data-generation)
- [3. Training](#3-training)


## 1. Environment Setup
```
conda env create -f ./environment.yml
```
## 2. Data Preparation
### Data Preprocessing
The 'data_preprocessing.py' file in the CICDDoS2019 folder outlines the preprocessing steps for the CICDDoS2019 dataset.
### Image Data Generation
Generate input using the code provided by the author of IGTD.
## 3. Training
We provide two example scripts for initiating a new training session and resuming a previous one. The best model will be saved in the `checkpoints/[model_name]/[model_name_with_configuration_and_time_info]/[model_name_epoch_loss_accuracy_best.pth]` directory.
```
# Start a new training session
python main.py -m SimpleViT --data-path path_to_your_dataset -nc 2 -is 7 -ps 7 \
-ch 1 -dim 8 -md 8 -dp 1 -heads 1 --cuda-id cuda_device_id --batch-size 64 -e \
100

# Resume the previous training
python main.py -m SimpleViT --data-path path_to_dataset -nc 2 -is 7 -ps 7 -ch \
1 -dim 8 -md 8 -dp 1 -heads 1 --cuda-id cuda_device_id --batch-size 64 -e 100 \
--continue-train True --saved-model-path path_to_saved_model
```
Available models for specification can be found in `parser/parser.py`.
<!-- ## 4. Testing -->