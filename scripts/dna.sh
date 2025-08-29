# VIDD
python "$(dirname "$0")/../finetune_reward.py" \
    --task dna \
    --wandb_mode online \
    --wandb_group dna_distillation \
    --wandb_name step128_KL_oldin5_lmbdas1_rxt_rgennew_lr1e-4 \
    --total_num_steps 128 \
    --loss_func KL \
    --gkd_lmbda 1.0 \
    --target_update_interval 5 \
    --learning_rate 1e-4 \
    --old_roll_in \
    --use_value_xt \
    --rs_gen_model new

# best of N
python "$(dirname "$0")/../finetune_reward.py" \
    --task dna \
    --wandb_mode online \
    --wandb_group dna_distillation \
    --wandb_name base_step128_bestof32 \
    --total_num_steps 128 \
    --best_of_n_baseline \
    --best_of_N 32


# standard fine-tuning
python "$(dirname "$0")/../finetune_reward.py" \
    --task dna \
    --wandb_mode online \
    --wandb_group dna_distillation \
    --wandb_name step128_KL_old1e6_lmbda0_lr1e-4 \
    --total_num_steps 128 \
    --loss_func KL \
    --gkd_lmbda 0 \
    --target_update_interval 100000 \
    --learning_rate 1e-4


# DDPO
python "$(dirname "$0")/../finetune_reward.py" \
    --task dna \
    --wandb_mode online \
    --wandb_group dna_distillation \
    --wandb_name step128_DDPO_old1_rnorm_lr1e-4_lmbda1 \
    --total_num_steps 128 \
    --loss_func DDPO \
    --target_update_interval 1 \
    --adv_norm \
    --learning_rate 1e-4 \
    --gkd_lmbda 1.0 \


# DDPP
python "$(dirname "$0")/../finetune_reward.py" \
    --task dna \
    --wandb_mode online \
    --wandb_group dna_distillation \
    --wandb_name step128_DDPP_alpha01_lr1e-4 \
    --total_num_steps 128 \
    --loss_func DDPP \
    --teacher_alpha 1.0 \
    --learning_rate 1e-4 \
    --gkd_lmbda 0.5

