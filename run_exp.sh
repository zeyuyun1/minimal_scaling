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




python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 2000 \
--lr 2e-4 --exp_name layer1-ff --model_arch recur_new --in_channels 1 --num_basis 128 \
--n_iters_inter 1 --n_iters_intra 1 --eta_base 0.2 \
--jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1 --sigma 0.01,0.4  \
--P_mean=-2 --P_std=0.5 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 2000 \
--lr 2e-4 --exp_name layer1-recur --model_arch recur_new --in_channels 1 --num_basis 128 \
--n_iters_inter 1 --n_iters_intra 1 --eta_base 0.1 \
--jfb_no_grad_iters 2,10 --jfb_with_grad_iters 1,3 --sigma 0.01,0.4  \
--P_mean=-2 --P_std=0.5 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop



# python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 500 \
# --lr 2e-4 --exp_name layer1-ae --model_arch AE --in_channels 1 --num_basis 128 \
# --sigma 0.01,0.4  \
# --P_mean=-2 --P_std=0.5 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
# --img_size 128 --random_crop

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 500 \
--lr 2e-4 --exp_name layer1-ae-whiten-noise-labels --model_arch AE --in_channels 1 --num_basis 128 \
--sigma 0.01,0.4  \
--P_mean=-2 --P_std=0.5 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 500 \
--lr 2e-4 --exp_name layer1-ae-whiten-noise-labels-mid-sigma --model_arch AE --in_channels 1 --num_basis 128 \
--sigma 0.01,1.0  \
--P_mean=-1.5 --P_std=0.8 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7



# python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
# --lr 2e-4 --exp_name layer1-ae-whiten-noise-labels --model_arch AE --in_channels 1 --num_basis 128 \
# --sigma 0.01,0.5  \
# --P_mean=-1.6 --P_std=0.7 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
# --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-ae-whiten-noise_cond --model_arch AE --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-ae-noise_cond --model_arch AE --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --kernel_size 7



CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-SC-whiten-noise-labels --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1 --no_learning_horizontal --eta_base 1.0


# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
# --lr 2e-4 --exp_name layer1-SC-whiten-noise-labels-load --model_arch SC --in_channels 1 --num_basis 128 \
# --sigma 0.01,0.5  \
# --P_mean=-1.6 --P_std=0.7 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
# --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
# --jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1 --no_learning_horizontal --eta_base 1.0 \
# --load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00033_layer1-SC-whiten-noise-labels/denoiser.ckpt


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-SC-whiten-noise-labels-recur --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 3,10 --jfb_with_grad_iters 1,3 --no_learning_horizontal --eta_base 0.1 \
--checkpoint_path pretrained_model/scaling-VH-new/00033_layer1-SC-whiten-noise-labels/denoiser.ckpt


CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-SC-whiten-noise-labels-recur --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 3,10 --jfb_with_grad_iters 1,3 --no_learning_horizontal --eta_base 0.1 \
--checkpoint_path pretrained_model/scaling-VH-new/00033_layer1-SC-whiten-noise-labels/denoiser.ckpt \
--load_all_weights --zero_M_weights \


#   python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
# --lr 2e-4 --exp_name layer1-SC-whiten-noise-labels-recur-scratch --model_arch SC --in_channels 1 --num_basis 128 \
# --sigma 0.01,0.5  \
# --P_mean=-1.6 --P_std=0.7 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
# --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
# --jfb_no_grad_iters 3,10 --jfb_with_grad_iters 1,3 --no_learning_horizontal --eta_base 0.1 \



python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 1 --kernel_size 7 \
--jfb_no_grad_iters 0,10 --jfb_with_grad_iters 1,3 --no_learning_horizontal --eta_base 0.1 \


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 1 --kernel_size 7 \
--jfb_no_grad_iters 0,10 --jfb_with_grad_iters 1,3 --no_learning_horizontal --eta_base 0.1 \

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3  --eta_base 0.1 \


CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-horizontal-weighting --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3  --eta_base 0.1 --edm_weighting \


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-weighting --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3  --eta_base 0.1 --edm_weighting --no_learning_horizontal \




CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-horizontal-weighting-prev --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3  --eta_base 0.1 --edm_weighting --jfb_reuse_solution \


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-horizontal-weighting-straighten --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 3,40 --jfb_with_grad_iters 1,5  --eta_base 0.1 --edm_weighting \
--load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00059_layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting/denoiser.ckpt



python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-horizontal-weighting-straighten --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 3,40 --jfb_with_grad_iters 1,5  --eta_base 0.1 --edm_weighting \
--load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00059_layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting/denoiser.ckpt



# stable experiment

CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-VH-stable-64 --batch_size 128 --n_epochs 1000 \
--lr 2e-4 --exp_name layer1-AE-whiten-noise-labels --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 64 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1  --eta_base 1.0 --edm_weighting --no_learning_horizontal \


python main.py --gpus 4 --project_name scaling-VH-stable-64 --batch_size 128 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-AE-whiten-simple-noise-labels-no-relu --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 64 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1  --eta_base 1.0 --edm_weighting --no_learning_horizontal --stable \


CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-VH-stable-64 --batch_size 128 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-simple-noise-labels-no-relu-full-ff --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 64 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 6,10 --jfb_with_grad_iters 1,3  --eta_base 1.0 --edm_weighting --no_learning_horizontal --stable \


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-ae-whiten-simple-noise-emb-edm-weight --model_arch SC --in_channels 1 \
--num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 \
--random_crop --whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1 \
--eta_base 0.1 --edm_weighting --no_learning_horizontal --stable





# stable round 2
python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-weighting-high-recur --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 5,10 --jfb_with_grad_iters 5,10  --eta_base 0.1 --edm_weighting --no_learning_horizontal \


# python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
# --lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-weighting-high-recur-reuse --model_arch SC --in_channels 1 --num_basis 128 \
# --sigma 0.01,0.5  \
# --P_mean=-1.6 --P_std=0.7 \
# --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
# --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
# --jfb_no_grad_iters 5,10 --jfb_with_grad_iters 5,10  --eta_base 0.1 --edm_weighting --no_learning_horizontal --jfb_reuse_solution \


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-weighting-high-recur-reuse-short --model_arch SC --in_channels 1 --num_basis 128 \
--sigma 0.01,0.5  \
--P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
--img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 \
--jfb_no_grad_iters 2,6 --jfb_with_grad_iters 1,2  --eta_base 0.25 --edm_weighting --no_learning_horizontal --jfb_reuse_solution \

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-noise-labels-zero-scratch-weighting-high-recur-reuse-long-horizontal-small-eta \
--model_arch SC --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 2,6 --jfb_with_grad_iters 1,3 --eta_base 0.1 \
--edm_weighting --jfb_reuse_solution



# reuse experiment

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.1 --edm_weighting \
--jfb_reuse_solution_rate 0.5

python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-no-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.1 --edm_weighting \
--jfb_reuse_solution_rate 0.


python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-straighten-from-no-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 4,20 --jfb_with_grad_iters 1,5 --eta_base 0.1 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00085_layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-no-reuse/denoiser.ckpt






# python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000  \
# --lr 1e-4 --exp_name layer1-ae-whiten-noise-emb-edm-weight --model_arch SC --in_channels 1  \
# --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ \
# --grayscale --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 0,0 \
# --jfb_with_grad_iters 1,1 --eta_base 1.0 --no_learning_horizontal


#  --edm_weighting
# --edm_weighting



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

