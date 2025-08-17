# VIDD
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_SS \
    --wandb_name RWMLE_rnormal_lr1e-5_lmbda1_oldin50_alpha1_rxt_rgennew_rstep_rclamp1e6 \
    --reward match_define_ss,plddt \
    --reward_weight 1,0 \
    --define_ss b \
    --batch_size 32 \
    --unmask_K 4 \
    --loss_func RW_MLE \
    --reward_norm normal \
    --learning_rate 1e-5 \
    --target_update_interval 50 \
    --old_roll_in \
    --gkd_lmbda 1.0 \
    --use_value_xt \
    --rs_gen_model new \
    --reward_step \
    --teacher_alpha 1.0


# best of N
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_SS \
    --wandb_name bestof32_base_ss_b_bs32_K4 \
    --reward match_define_ss \
    --reward_weight 1 \
    --define_ss b \
    --batch_size 32 \
    --unmask_K 4 \
    --best_of_n_baseline \
    --best_of_N 32


# standard fine-tuning
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_SS \
    --wandb_name ss_b_bs32_K4_rnormal_lr1e-5_lmbda0_old1e6 \
    --reward match_define_ss,plddt \
    --reward_weight 1,0 \
    --define_ss b \
    --batch_size 32 \
    --unmask_K 4 \
    --loss_func KL \
    --reward_norm normal \
    --learning_rate 1e-5 \
    --target_update_interval 100000 \
    --gkd_lmbda 0 \


# DDPO
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_SS \
    --wandb_name DDPO_bs32_K4_old1rollin_rnorm_lr1e-5_lmbda1 \
    --reward match_define_ss,plddt \
    --reward_weight 1,0 \
    --define_ss b \
    --batch_size 32 \
    --unmask_K 4 \
    --loss_func DDPO \
    --target_update_interval 1 \
    --adv_norm \
    --learning_rate 1e-5 \
    --gkd_lmbda 1.0 \
    --old_roll_in



# DDPP
python "$(dirname "$0")/../finetune_reward_protein.py" \
    --wandb_mode online \
    --wandb_group protein_distillation_SS \
    --wandb_name DDPP_ss_b_bs32_K4_alpha1_lr1e-5_lmbda1 \
    --reward match_define_ss,plddt \
    --reward_weight 1,0 \
    --define_ss b \
    --batch_size 32 \
    --unmask_K 4 \
    --learning_rate 1e-5 \
    --loss_func DDPP \
    --teacher_alpha 1 \
    --gkd_lmbda 1.0

