python train.py \
    --data_path data/bbbp.csv \
    --dataset_type classification \
    --epochs 90 \
    --clip_norm 5.0 \
    --lam 1 \
    --topk 0.5 \
    --hidden_size 240 \
    --depth 2 \
    --dropout 0.4 \
    --activation LeakyReLU \
    --ffn_num_layers 3 \
    --ffn_hidden_size 240 \
    --batch_size 96 \
    --max_lr 0.0009 \
    --init_lr 0.0001 \
    --final_lr 5e-05 \
    --adv_w 1.0 \
    --beta 0.9 \
    --eps_max 0.07 \
    --atten_head 4 \
    --atten_dropout 0.3


python train.py \
    --data_path data/bace.csv \
    --dataset_type classification \
    --epochs 90 

python train.py \
    --data_path data/tox21.csv \
    --dataset_type classification \
    --epochs 90 


python train.py \
    --data_path data/sider.csv \
    --dataset_type classification \
    --epochs 90 

python train.py \
    --data_path data/esol.csv \
    --dataset_type regression \
    --epochs 90 



python train.py \
    --data_path data/freesolv.csv \
    --dataset_type regression \
    --epochs 100 \
    --clip_norm 5.0 \
    --topk 0.3 \
    --hidden_size 480 \
    --depth 2 \
    --lam 0.2 \
    --dropout 0.5 \
    --activation LeakyReLU \
    --ffn_num_layers 1 \
    --ffn_hidden_size 480 \
    --batch_size 64 \
    --max_lr 0.0025 \
    --init_lr 0.0003 \
    --final_lr 0.0025 \
    --adv_w 0.6 \
    --eps_max 0.9 \
    --atten_head 8 \
    --atten_dropout 0.0


python train.py     --data_path data/lipo.csv \
    --dataset_type regression \
    --epochs 95 \
    --clip_norm 5.0 \
    --lam 1 \
    --topk 0.6 \
    --hidden_size 480 \
    --depth 3 \
    --dropout 0.1 \
    --activation LeakyReLU \
    --ffn_num_layers 1 \
    --ffn_hidden_size 360 \
    --batch_size 64 \
    --max_lr 0.00056 \
    --init_lr 6.7e-05 \
    --final_lr 0.003 \
    --adv_w 0.5 \
    --eps_max 0.6 \
    --atten_head 6 \
    --atten_dropout 0.6


python train.py     --data_path data/clintox.csv \
    --dataset_type classification \
    --epochs 90 \
    --clip_norm 5.0 \
    --lam 1 \
    --topk 0.6 \
    --hidden_size 360 \
    --depth 2 \
    --dropout 0.1 \
    --activation LeakyReLU \
    --ffn_num_layers 1 \
    --ffn_hidden_size 120 \
    --batch_size 96 \
    --max_lr 0.00136 \
    --init_lr 5e-05 \
    --final_lr 7.4e-05 \
    --adv_w 0.9 \
    --eps_max 0.9 \
    --atten_head 4 \
    --atten_dropout 0.2


