# AAIM
This is the code for our work: Adaptive and Asynchronous Integration of Gray and White Matter fMRI for Brain Disorder Diagnosis

# Quick Start
1. Configure the dependencies in requirements.txt.
2. Train the model：
python train.py --dataset 'ADNI' --GM_n 90 --WM_n 50 --config '[[8, 8, 2, 6],[8, 8, 2, 6],[8, 8, 2, 6]]' --modu_coef 0.4 --lr 0.1 --lr_decay --end_epoch 200


