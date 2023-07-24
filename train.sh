export CUDA_VISIBLE_DEVICES=0,1,2,3
#for first stage training (L1 Model)
bash tools/dist_train.sh configs/egovsr/egovsr_L1_reds.py 4
#for second stage training (GAN Model)
bash tools/dist_train.sh configs/egovsr/egovsr_reds.py 4