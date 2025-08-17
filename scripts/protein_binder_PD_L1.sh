# VIDD
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_PDL1 \
    --wandb_name RWMLE_K4_i1p01r002_rnormal_lr1e-5_lmbda08_oldin50_alpha1_rxt_rgennew_rstep \
    --reward iptm,plddt,radius \
    --reward_weight 1,0.1,0.02 \
    --bind_target PDL1 \
    --batch_size 32 \
    --unmask_K 4 \
    --gen_len 100 \
    --loss_func RW_MLE \
    --reward_norm normal \
    --learning_rate 1e-5 \
    --target_update_interval 50 \
    --old_roll_in \
    --gkd_lmbda 0.8 \
    --teacher_alpha 1.0 \
    --use_value_xt \
    --rs_gen_model new \
    --reward_step

# best of N
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_PDL1 \
    --wandb_name bestof128_i1p01r002_base_bs32_K4 \
    --reward iptm,plddt,radius \
    --reward_weight 1,0.1,0.02 \
    --bind_target PDL1 \
    --batch_size 32 \
    --unmask_K 4 \
    --best_of_n_baseline \
    --best_of_N 128 \
    --gen_len 100


# standard fine tuning
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_PDL1 \
    --wandb_name stand_i1p01r002_rnormal_lr1e-5_lmbda0_old1e6 \
    --reward iptm,plddt,radius \
    --reward_weight 1,0.1,0.02 \
    --bind_target PDL1 \
    --batch_size 32 \
    --unmask_K 4 \
    --gen_len 100 \
    --loss_func RW_MLE \
    --reward_norm normal \
    --learning_rate 1e-5 \
    --target_update_interval 100000 \
    --gkd_lmbda 0



# DDPO
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_PDL1 \
    --wandb_name DDPO_i1p01r002_bs32_K4_old1rollin_rnorm_lr1e-5_lmbda1 \
    --reward iptm,plddt,radius \
    --reward_weight 1,0.1,0.02 \
    --batch_size 32 \
    --unmask_K 4 \
    --loss_func DDPO \
    --target_update_interval 1 \
    --adv_norm \
    --learning_rate 1e-5 \
    --gkd_lmbda 1.0 \
    --old_roll_in \
    --gen_len 100


# DDPP
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_PDL1 \
    --wandb_name DDPP_bs32_K4_alpha1_lr1e-5_lmbda1 \
    --reward iptm,plddt,radius \
    --reward_weight 1,0.1,0.02 \
    --bind_target PDL1 \
    --batch_size 32 \
    --unmask_K 4 \
    --learning_rate 1e-5 \
    --loss_func DDPP \
    --teacher_alpha 1 \
    --gkd_lmbda 1.0 \
    --gen_len 100





