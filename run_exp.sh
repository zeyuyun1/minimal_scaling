# conda activate denoise-ddp \
# python main.py \
# --gpus 4 \
# --project_name scaling-VH \
# --batch_size 64 \
# --n_epochs 1000 \
# --lr 2e-4 \
# --exp_name layer3-big  \
# --in_channels 1 \
# --num_basis 128,128,128 \
# --n_iters_inter 1 \
# --n_iters_intra 4 \
# --eta_base 0.5 \
# --sigma 0.01,1 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ \
# --grayscale \
# --img_size 128 \
# --random_crop


# --dataset general \
# --no_normalize \
# --no_resize \


# python main.py \
# --gpus 4 \
# --project_name scaling-new \
# --batch_size 128 \
# --n_epochs 1000 \
# --lr 2e-4 \
# --exp_name layer4-edm-predX  \
# --in_channels 3 \
# --num_basis 128,128,128,128 \
# --n_iters_inter 1 \
# --n_iters_intra 4 \
# --eta_base 0.5 \
# --sigma 0.01,2 \
# --data_dir /home/zeyu/celeba/ \
# --img_size 64 \
# # --grayscale \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-base \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 4 --eta_base 0.5 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-base-no-recur \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.5 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-base-edm-weighting \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 4 --eta_base 0.5 --sigma 0.01,2 --edm_weighting \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-base-low-noise \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 4 --eta_base 0.5 --sigma 0.01,2 --P_mean -2 --P_std 0.5 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-recur \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 2 --n_iters_intra 4 --eta_base 0.5 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-ff-new --model_arch recur_new \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.5 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-ff-new --model_arch recur_new \
# --in_channels 3 --num_basis 64,64,128,128,128 \
# --n_iters_inter 1 --n_iters_intra 4 --eta_base 0.5 --sigma 0.01,2 \

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
--batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-recur-no-intra \
--in_channels 3 --num_basis 64,64,128,128,128 \
--n_iters_inter 2 --n_iters_intra 1 --eta_base 0.5 --sigma 0.01,2 \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-recur-no-intra-smaller \
# --in_channels 3 --num_basis 64,64,128,128 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.15 --sigma 0.01,2 \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer5-recur-no-intra-smaller \
# --in_channels 3 --num_basis 64,64,128,128 \
# --n_iters_inter 2 --n_iters_intra 4 --eta_base 0.15 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer2-intra-deq-small-eta --model_arch recur_new \
# --in_channels 3 --num_basis 64,64 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.1 --sigma 0.01,2


# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer2-no-recur \
# --in_channels 3 --num_basis 64,64 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.2 --sigma 0.01,2 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --gpus 8 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer4-intra-deq-small-eta --model_arch recur_new \
# --in_channels 3 --num_basis 64,64,128,128 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_base 0.01 --sigma 0.01,2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --gpus 8 --project_name scaling-new-face \
# --batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer4-intra-deq-multi-eta --model_arch recur_new \
# --in_channels 3 --num_basis 64,64,128,128 \
# --n_iters_inter 1 --n_iters_intra 1 --eta_ls 0.01,0.05,0.01,0.005 --sigma 0.01,2


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --gpus 8 --project_name scaling-new-face \
--batch_size 128 --n_epochs 400 --lr 2e-4 --exp_name layer4-intra-deq-multi-eta --model_arch recur_new \
--in_channels 3 --num_basis 64,64,128,128 \
--n_iters_inter 1 --n_iters_intra 1 --eta_ls 0.1,0.1,0.05,0.02 --sigma 0.01,2 \
--jfb_no_grad_iters 4,10 --jfb_with_grad_iters 1,2 --jfb_reuse_solution --jfb_ddp_safe