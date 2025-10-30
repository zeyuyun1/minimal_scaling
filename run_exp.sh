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

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-straighten-from-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 4,20 --jfb_with_grad_iters 1,5 --eta_base 0.1 --edm_weighting \
--jfb_reuse_solution_rate 0.5 --load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00086_layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-reuse/denoiser.ckpt
 


# straighten non horizontal model
python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 --lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-weighting-straighten \
 --model_arch SC --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 --data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale \
  --img_size 128 --random_crop --whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 5,20 --jfb_with_grad_iters 1,5 --eta_base 0.1 --edm_weighting --no_learning_horizontal \
 --load_all_weights --checkpoint_path pretrained_model/scaling-VH-new/00061_layer1-SC-whiten-1-noise-labels-zero-scratch-weighting/denoiser.ckpt





# Group Conv experiment
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-horizontal-weighting-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.1 --edm_weighting \
--jfb_reuse_solution_rate 0.5




# reuse + Safe + Smaller learning rate

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-reuse-no-whiten \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --no_learning_horizontal



python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --no_learning_horizontal



CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-reuse-fgroups-4 \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --no_learning_horizontal --frequency_groups 4


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-reuse-fgroups-4 \
--model_arch SC_reuse --in_channels 1 --num_basis 256 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --no_learning_horizontal --frequency_groups 8



# extreme
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1-noise-labels-zero-scratch-reuse-fgroups-2 \
--model_arch SC_reuse --in_channels 1 --num_basis 256 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --no_learning_horizontal --frequency_groups 2








python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 \
--n_epochs 1000 --lr 1e-4 --exp_name layer1-AE-whiten-1-noise-labels-zero-scratch-reuse \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 16 --kernel_size 7 --jfb_no_grad_iters 0,0 --jfb_with_grad_iters 1,1 --eta_base 1 \
--edm_weighting --jfb_reuse_solution_rate 0.0 --no_learning_horizontal


python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-1real-noise-labels-zero-scratch-reuse-horizontal \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 1 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3



CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-VH-new-2 --batch_size 64 --n_epochs 1000 \
--lr 1e-4 --exp_name layer1-SC-whiten-4-noise-labels-zero-scratch-reuse-horizontal-white-kernel-7 \
--model_arch SC_reuse --in_channels 1 --num_basis 128 --sigma 0.01,0.5 --P_mean=-1.6 --P_std=0.7 \
--data_dir /home/zeyu/vanhateren_all/vh_patches256_train/ --grayscale --img_size 128 --random_crop \
--whiten_dim 4 --kernel_size 7 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 --eta_base 0.05 --edm_weighting \
--jfb_reuse_solution_rate 0.3 --whiten_ks 7


# MNIST
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-white --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,2.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --whiten_dim 4 --kernel_size 7 
# --data_dir /home/zeyu/mnist/ --img_size 28 --random_crop

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-no-white-zero_emb --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,2.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7 


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-white --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,2.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --whiten_dim 4 --kernel_size 7 


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-white-no-white-zero_simple_emb --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,2.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7 

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-white-no-white-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,1.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1

python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-horizontal-white-1-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,1.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --whiten_dim 1 --whiten_ks 5


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-white-1-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,1.0 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_learning_horizontal --whiten_dim 1 --whiten_ks 5



# Buxinle
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.05 --sigma 0.01,0.5 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_learning_horizontal --whiten_dim 2 --whiten_ks 5




CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 64 --eta_base 0.01 --sigma 0.1,1.0 --jfb_no_grad_iters 10,15 --jfb_with_grad_iters 1,5 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_learning_horizontal

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-zero_simple_emb_sigma_small-horizontal --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.005 --sigma 0.1,1.0 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,5 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_noise_embedding

python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.001 --sigma 0.1,1.0 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,5 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_learning_horizontal --no_noise_embedding 


# 1. low noise regime, sparse coding vs SAE

python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_small-horizontal --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.05 --sigma 0.01,0.3 --jfb_no_grad_iters 1,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7  --no_noise_embedding --loss_type uniform


python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.01 --sigma 0.1,0.1 --jfb_no_grad_iters 20,40 --jfb_with_grad_iters 1,6 \
--dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias


python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_small --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.02 --sigma 0.2,0.2 --jfb_no_grad_iters 70,100 --jfb_with_grad_iters 1,5 \
--dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 3e-4 --exp_name layer-1-AE-zero_simple_emb_sigma_small-no-bias --model_arch AE --in_channels 1 \
--num_basis 128  --sigma 0.2,0.2  \
--dataset mnist --kernel_size 7 --no_noise_embedding --loss_type uniform --no_bias 


# # multi-noise
# python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
# --n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_large --model_arch SC_reuse --in_channels 1 \
# --num_basis 128 --eta_base 0.02 --sigma 0.01,1.0 --jfb_no_grad_iters 35,50 --jfb_with_grad_iters 1,5 \
# --dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias


# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
# --n_epochs 300 --lr 3e-4 --exp_name layer-1-AE-zero_simple_emb_sigma_small-no-bias_sigma_large --model_arch AE --in_channels 1 \
# --num_basis 128  --sigma 0.2,0.2  \
# --dataset mnist --kernel_size 7 --no_noise_embedding --loss_type uniform --no_bias


# python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
# --n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_mid --model_arch SC_reuse --in_channels 1 \
# --num_basis 128 --eta_base 0.02 --sigma 0.5,0.5 --jfb_no_grad_iters 35,50 --jfb_with_grad_iters 1,5 \
# --dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias


python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_mid_scale --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.02 --sigma 0.5,0.5 --jfb_no_grad_iters 35,50 --jfb_with_grad_iters 1,5 \
--dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias


python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_mid_horizontal_fast_bias --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.02 --sigma 0.5,0.5 --jfb_no_grad_iters 2,10 --jfb_with_grad_iters 1,3 \
--dataset mnist --kernel_size 7 --no_noise_embedding --loss_type uniform --no_bias

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 \
--n_epochs 300 --lr 3e-4 --exp_name layer-1-AE-zero_simple_emb_sigma_mid-no-bias --model_arch AE --in_channels 1 \
--num_basis 128  --sigma 0.5,0.5  \
--dataset mnist --kernel_size 7 --no_noise_embedding --loss_type uniform --no_bias



python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 --n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_white_ks_1_dim_1 --model_arch SC_reuse --in_channels 1 --num_basis 128 --eta_base 0.01 --sigma 0.5,0.5 --jfb_no_grad_iters 35,50 --jfb_with_grad_iters 1,5 --dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias --whiten_dim 2 --whiten_ks 1

 python main.py --gpus 4 --project_name scaling-new-mnist --batch_size 128 --n_epochs 300 --lr 1e-4 --exp_name layer-1-SC-zero_simple_emb_sigma_white_ks_1_dim_2 --model_arch SC_reuse --in_channels 1 --num_basis 128 --eta_base 0.015 --sigma 0.5,0.5 --jfb_no_grad_iters 70,100 --jfb_with_grad_iters 1,10 --dataset mnist --kernel_size 7 --no_learning_horizontal --no_noise_embedding --loss_type uniform --no_bias --whiten_dim 2 --whiten_ks 1








# Let's try again on MNIST
# all noise level 0 - 1.0, sparse autoencoder vs sparse coding vs sparse coding with HC
# Prediction: 
# sparse coding doesn't work at all because you shouldn't reconstruct noise.
# sparse coding with HC should work the best. But the excersice is to figure out why it works [When does the model reconstruct? When does the model stop reconstruct, etc.]
# How about no noise embedding for all models?


# new experiment
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --gpus 4 --project_name scaling-mnist-exercise --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-SC-horizontal-large-noise --model_arch SC_reuse --in_channels 1 \
--num_basis 128 --eta_base 0.02 --sigma 0.01,2.0 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,4 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1


python main.py --gpus 4 --project_name scaling-mnist-exercise --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-SC-horizontal-small-noise-no-NE-0.2 --model_arch SC_reuse --in_channels 1 \
--num_basis 32 --eta_base 0.2 --sigma 0.01,0.1 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,4 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1 --no_learning_horizontal --no_noise_embedding --no_bias


python main.py --gpus 4 --project_name scaling-mnist-exercise --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-SC-horizontal-mid-noise-no-NE --model_arch SC_reuse --in_channels 1 \
--num_basis 32 --eta_base 0.1 --sigma 0.01,0.5 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,4 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1  --no_noise_embedding --no_bias



CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-mnist-exercise --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-SC-horizontal-mid-0.3-noise-no-NE --model_arch SC_reuse --in_channels 1 \
--num_basis 32 --eta_base 0.02 --sigma 0.01,0.3 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,4 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1  --no_noise_embedding 


CUDA_VISIBLE_DEVICES=4,5,6,7  python main.py --gpus 4 --project_name scaling-mnist-exercise --batch_size 128 \
--n_epochs 1000 --lr 1e-4 --exp_name layer-1-SC-mid-0.3-noise-no-NE --model_arch SC_reuse --in_channels 1 \
--num_basis 32 --eta_base 0.02 --sigma 0.01,0.3 --jfb_no_grad_iters 5,15 --jfb_with_grad_iters 1,4 \
--dataset mnist --kernel_size 7  --P_mean=-1.4 --P_std=1.1  --no_noise_embedding --no_learning_horizontal







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

