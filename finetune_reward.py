import argparse
import wandb
import numpy as np
import torch
import random
from datetime import datetime
import torch.nn.functional as F
import os, copy
from tqdm import tqdm
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from evaluations.dna_eval import batch_agg_count_k_mers
from losses.rl_loss import *
from models.gen_models import initialize_gen_model, generate_xt_list
from diffusion_gosai import _sample_categorical
import diffusion_gosai
from evaluations.dna_eval import DNAEvalMetrics
from dataloader_gosai import batch_dna_tokenize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_diffuse(model, x0, t):
    sigma_t, _ = model.noise(t)
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_t = move_chance_t[:, None]  # shape [B, 1]
    xt = model.q_xt(x0, move_chance_t)
    return xt, move_chance_t


def get_random_t(T, batch_size, device):
    """
    Sample a random timestep t in [0, T-1] for each item in the batch.

    Args:
        T (int): Total number of diffusion steps.
        batch_size (int): Number of data points in the batch.
        device (torch.device): Device for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing random integers in [0, T-1].
    """
    return torch.randint(low=0, high=T, size=(batch_size,), device=device)


def read_fasta(fasta_file):
    sequences = []
    with open(fasta_file, "r") as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences


def run(args, rank=None):
    # initialization
    set_seed(args.seed)
    args_dict = vars(args)
    wandb.init(
        id=f"{args.wandb_name}+{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        group=args.wandb_group,
        mode=args.wandb_mode,
        project="Distillation",
        job_type='FA',
        name=args.wandb_name,
        # track hyperparameters and run metadata
        config=args_dict,
    )

    # save address
    result_save_folder = os.path.join('output', f"{args.task}_{args.reward}_{args.wandb_name}")
    save_folders_model = os.path.join(result_save_folder, "models")
    os.makedirs(save_folders_model, exist_ok=True)
    # save_folders_log = os.path.join(result_save_folder, "logs")
    # os.makedirs(save_folders_log, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """initialize diffusion model & reward model"""
    model_collections = initialize_gen_model(args, device)
    pre_model, old_model, new_model = model_collections['pre_model'], model_collections['old_model'], model_collections['new_model']
    eval_models = DNAEvalMetrics(
        base_path=args.data_base_path,
        device=device,
    )

    # prepare dataset
    all_sequences = None
    if args.use_decoding:
        all_sequences = read_fasta(args.decoding_save_path)

    # diffusion steps
    # for protein evodiff, timestep = length (unmask each token at one time)
    timesteps = torch.linspace(1, args.eps, args.total_num_steps + 1, device=device)  # [129,200]
    dt = (1 - args.eps) / args.total_num_steps

    if args.only_test:
        result_dict = best_of_n_test(args=args, eval_models=eval_models, timesteps=timesteps, dt=dt, device=device,
                                     best_model_path=args.test_model_path, pre_model=pre_model)
        print(result_dict)
        return

    if args.svdd_baseline:
        svdd_result_dict = svdd_test(args=args, eval_models=eval_models, timesteps=timesteps, dt=dt, device=device)
        wandb.log(svdd_result_dict)
        wandb.finish()
        return
    if args.best_of_n_baseline:
        result_dict = best_of_n_test(args=args, eval_models=eval_models, timesteps=timesteps, dt=dt, device=device, pre_model=pre_model)
        wandb.log(result_dict)
        wandb.finish()
        return

    new_model.train()
    value_net = None
    log_fs = None

    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)

    best_rewards_eval = float('-inf')
    best_rewards, best_atac_acc_rewards, best_k_mer = float('-inf'), float('-inf'), float('-inf')
    best_rewards_mid_std, best_rewards_eval_std, best_atac_acc_rewards_std = float('-inf'), float('-inf'), float('-inf')
    best_model_path = os.path.join(save_folders_model, f'model_best.ckpt')
    for epoch_num in tqdm(range(args.num_epochs), desc="training epochs"):
        rewards, rewards_eval = [], []  # take median
        atac_acc_rewards = []  # take mean
        k_mer_count_dict, k_mer_length = {}, 0
        motif_corr_list = []
        model_logl_list = []  # take median
        losses = []
        all_reward_weights, all_log_probs, all_kl, all_pred_value = [], [], [], []
        # reward_losses = []
        # kl_losses = []
        tot_grad_norm = 0.0
        # record previous new model params
        prev_new_model = copy.deepcopy(new_model.state_dict())
        for _step in range(args.num_accum_steps):
            with torch.no_grad():
                """generate dataset"""
                sample_new, last_x_list_new, condt_list_new, conds_list_new, \
                    move_chance_t_list_new, move_chance_s_list_new, copy_flag_list_new = \
                    generate_xt_list(
                        args=args,
                        generative_model=new_model,
                        device=device,
                        timesteps=timesteps,
                        dt=dt,
                        reward_model=eval_models,
                        num_best_of_N=args.best_of_N,
                        train_stage=True,
                    )

                if args.use_schedule_rollin:
                    gkd_lmbda = epoch_num / (args.num_epochs * args.gkd_lmbda)
                    # args.gkd_lmbda low -> more quick to student model
                    # args.gkd_lmbda high -> always pretrain
                else:
                    gkd_lmbda = args.gkd_lmbda

                if random.random() < gkd_lmbda:
                    # student model as roll-in distribution
                    last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                        last_x_list_new, condt_list_new, conds_list_new, move_chance_t_list_new, move_chance_s_list_new, copy_flag_list_new
                    current_roll_in = "student"
                    final_sample = sample_new
                else:
                    if args.use_decoding:
                        sample_pre_str = random.sample(all_sequences, args.batch_size)
                        sample_pre = batch_dna_tokenize(sample_pre_str)
                        sample_pre = torch.tensor(sample_pre, device=sample_new.device, dtype=sample_new.dtype)
                    else:
                        sample_pre, last_x_list_pre, condt_list_pre, conds_list_pre, \
                            move_chance_t_list_pre, move_chance_s_list_pre, copy_flag_list_pre = \
                            generate_xt_list(
                                args=args,
                                generative_model=pre_model,
                                device=device,
                                timesteps=timesteps,
                                dt=dt,
                                reward_model=eval_models,
                                num_best_of_N=args.best_of_N,
                                train_stage=True,
                            )

                        # classifier free fine-tuning
                        last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                            last_x_list_pre, condt_list_pre, conds_list_pre, move_chance_t_list_pre, move_chance_s_list_pre, copy_flag_list_pre
                    current_roll_in = "pretrain"
                    final_sample = sample_pre

            """observe performances of student generated samples"""
            # Pred-Activity, rewards_eval is more important, higher better
            reward_argmax = eval_models.cal_pred_activity(sample_new, reward_mode="train")  # bs
            reward_argmax_eval = eval_models.cal_pred_activity(sample_new, reward_mode="eval")  # bs
            rewards.append(reward_argmax.detach().cpu().numpy())
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            # ATAC-Acc, higher better
            atac_reward = eval_models.cal_atac_acc(sample_new)  # bs
            atac_acc_rewards.append(atac_reward > 0.5)
            # k-mer
            k_mer_count_dict, k_mer_length = batch_agg_count_k_mers(
                dna_seqs=sample_new.detach().cpu().numpy(),
                counts=k_mer_count_dict,
                length=k_mer_length,
            )
            # motif corr
            # motif_corr = eval_models.motif_corr(sample_new.cpu().numpy())
            # motif_corr_list.append(motif_corr)
            # log likelihood
            model_logl = pre_model.get_likelihood(
                x0=sample_new,
                num_steps=args.total_num_steps,
                eps=args.eps,
                n_samples=1,
            )  # bs
            model_logl_list.append(model_logl.detach().cpu().numpy())

            total_num_steps = range(args.total_num_steps)

            """model training"""
            new_model.train()
            total_loss, total_reward_weight, total_log_probs = [], [], []
            total_kl, total_value = [], []
            total_value_loss = []
            for random_t in tqdm(total_num_steps, desc="timestep iteration", disable=True):
                if current_roll_in == 'pretrain' and args.use_decoding:
                    # t_indices = get_random_t(args.total_num_steps, B, device=device)
                    t = timesteps[random_t].expand(args.batch_size)  # [B, 1]
                    ts = timesteps[random_t - 1] if random_t > 0 else None
                    xt = forward_diffuse(new_model, sample_pre, t.to(device))[0]
                    xs = forward_diffuse(new_model, sample_pre, t.to(device))[0] if ts is not None else sample_pre
                    xt = F.one_hot(xt, num_classes=new_model.vocab_size).to(torch.float32).detach()
                    xs = F.one_hot(xs, num_classes=new_model.vocab_size).to(torch.float32).detach()

                    # Model prediction log p(x0 | xt)
                    sigma_t, _ = new_model.noise(t.to(device))
                    sigma_s, _ = new_model.noise(t.to(device) - dt)
                    if sigma_t.ndim > 1:
                        sigma_t = sigma_t.squeeze(-1)
                    if sigma_s.ndim > 1:
                        sigma_s = sigma_s.squeeze(-1)
                    assert sigma_t.ndim == 1, sigma_t.shape
                    assert sigma_s.ndim == 1, sigma_s.shape
                    move_chance_t = 1 - torch.exp(-sigma_t)
                    move_chance_s = 1 - torch.exp(-sigma_s)
                    move_chance_t = move_chance_t[:, None, None]
                    move_chance_s = move_chance_s[:, None, None]
                    condt, conds = sigma_t, sigma_s
                else:
                    xt = last_x_list[random_t]  # dna: [bsz, seqlen, 5]. protein: bs * len
                    xs = last_x_list[random_t + 1] if random_t < args.total_num_steps - 1 else None
                    condt = condt_list[random_t]
                    conds = conds_list[random_t]
                    move_chance_t = move_chance_t_list[random_t]
                    move_chance_s = move_chance_s_list[random_t]
                    # copy_flag = copy_flag_list[random_t]  # [bsz, seqlen, 1], x_t is not mask: 1, x_t is mask: 0

                xt_seq = xt.argmax(dim=-1)  # [bsz, seqlen]
                loss_dict = dna_loss(
                    args=args,
                    new_model=new_model,
                    old_model=old_model,
                    pre_model=pre_model,
                    eval_models=eval_models,
                    xt=xt,
                    xt_seq=xt_seq,
                    condt=condt,
                    conds=conds,
                    move_chance_t=move_chance_t,
                    move_chance_s=move_chance_s,
                    # copy_flag=copy_flag,
                    current_roll_in=current_roll_in,
                    value_net=value_net,
                    progress_t=random_t,
                    final_reward=reward_argmax,
                    sample_new=final_sample,
                    log_fs=log_fs,
                    xs_define=xs,
                )
                reward_weight, log_probs, loss = loss_dict['reward_weight'], loss_dict['log_probs'], loss_dict['loss']
                if 'kl_term' in loss_dict:
                    kl_term = loss_dict['kl_term']
                    total_kl.append(kl_term)
                if 'value' in loss_dict:
                    pred_value = loss_dict['value']
                    total_value.append(pred_value)

                if loss is not None:
                    total_loss.append(loss)
                if reward_weight is not None:
                    total_reward_weight.append(reward_weight)
                if log_probs is not None:
                    total_log_probs.append(log_probs)

            loss = torch.stack(total_loss).mean()
            reward_weight = torch.stack(total_reward_weight).mean()
            log_probs = torch.stack(total_log_probs).mean()
            kl = torch.stack(total_kl).mean() if len(total_kl) > 0 else None
            pred_value = torch.stack(total_value).mean() if len(total_value) > 0 else None

            # loss = torch.stack(total_loss, 1).sum(1).mean()
            loss = loss / args.num_accum_steps
            reward_weight = reward_weight / args.num_accum_steps
            log_probs = log_probs / args.num_accum_steps
            kl = kl / args.num_accum_steps if kl is not None else None
            pred_value = pred_value / args.num_accum_steps if pred_value is not None else None

            loss.backward()
            if (_step + 1) % args.num_accum_steps == 0:  # Gradient accumulation
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.grad_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            all_reward_weights.append(reward_weight.cpu().detach().numpy() * args.num_accum_steps)
            all_log_probs.append(log_probs.cpu().detach().numpy() * args.num_accum_steps)
            if kl is not None:
                all_kl.append(kl.cpu().detach().numpy() * args.num_accum_steps)
            if pred_value is not None:
                all_pred_value.append(pred_value.cpu().detach().numpy() * args.num_accum_steps)
            # if args.loss_func == 'GRPO':
            #     reward_losses.append(avg_reward_loss.cpu().detach().numpy())
            #     kl_losses.append(avg_kl_loss.cpu().detach().numpy())

        if args.target_update_interval > 0 and epoch_num % args.target_update_interval == 0 and epoch_num != 0:
            # update theta_old by theta
            if args.loss_func == 'DDPO':
                old_model.load_state_dict(prev_new_model)
            else:
                old_model.load_state_dict(new_model.state_dict())

        # evaluation performances
        result_dict = {}
        losses = np.array(losses)
        reward_weights = np.array(all_reward_weights)
        log_probs = np.array(all_log_probs)
        kl = np.array(all_kl)
        pred_value = np.array(all_pred_value)

        rewards_mid = np.median(np.concatenate(rewards)).item()
        rewards_mid_eval = np.median(np.concatenate(rewards_eval)).item()
        atac_acc_rewards_mean = np.mean(np.concatenate(atac_acc_rewards)).item()
        k_mer_value = eval_models.compare_kmer(
            kmer2=k_mer_count_dict,
            n_sp2=k_mer_length,
        ).item()
        rewards_mid_std = np.std(np.concatenate(rewards)).item()
        rewards_eval_std = np.std(np.concatenate(rewards_eval)).item()
        atac_acc_rewards_std = np.std(np.concatenate(atac_acc_rewards)).item()
        logl_mid = np.median(np.concatenate(model_logl_list)).item()
        logl_std = np.std(np.concatenate(model_logl_list)).item()

        cur_result_dict = {
            "median_reward": rewards_mid,
            "median_reward_eval": rewards_mid_eval,
            "mean_atac_acc": atac_acc_rewards_mean,
            "k_mer": k_mer_value,
            "mean_grad_norm": tot_grad_norm,
            "std_reward": rewards_mid_std,
            "std_reward_eval": rewards_eval_std,
            "std_atac_acc": atac_acc_rewards_std,
            "median_logl": logl_mid,
            "std_logl": logl_std,
        }

        loss_dict = {}
        if len(losses) > 0:
            loss_dict['mean_loss'] = np.mean(losses)
        if len(reward_weights) > 0:
            loss_dict['mean_reward_weights'] = np.mean(reward_weights)
        if len(log_probs) > 0:
            loss_dict['mean_log_probs'] = np.mean(log_probs)
        if len(kl) > 0:
            loss_dict['mean_kl'] = np.mean(kl)
        if len(pred_value) > 0:
            loss_dict['mean_pred_value'] = np.mean(pred_value)

        result_dict.update(cur_result_dict)
        result_dict.update(loss_dict)
        # print(result_dict)
        wandb.log(result_dict, step=epoch_num)

        # save model
        if (epoch_num + 1) % args.save_every_n_epochs == 0 and args.wandb_name != 'debug':
            model_path = os.path.join(save_folders_model, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
        # save best model
        if rewards_mid > best_rewards:
        # if rewards_mid_eval > best_rewards_eval:
            best_rewards_eval = rewards_mid_eval
            best_rewards, best_atac_acc_rewards, best_k_mer = rewards_mid, atac_acc_rewards_mean, k_mer_value
            best_rewards_mid_std, best_rewards_eval_std, best_atac_acc_rewards_std = rewards_mid_std, rewards_eval_std, atac_acc_rewards_std
            torch.save(new_model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch_num}")

    wandb.log({
        "final_median_reward": best_rewards,
        "final_median_reward_eval": best_rewards_eval,
        "final_mean_atac_acc": best_atac_acc_rewards,
        "final_k_mer": best_k_mer,
        "final_std_reward": best_rewards_mid_std,
        "final_std_reward_eval": best_rewards_eval_std,
        "final_std_atac_acc": best_atac_acc_rewards_std,
    })

    wandb.finish()


def svdd_test(
        args,
        eval_models,
        timesteps, dt, device, best_model_path=None,
):
    GlobalHydra.instance().clear()
    # Initialize Hydra and compose the configuration
    initialize(config_path="configs_gosai", job_name="load_model")
    cfg = compose(config_name="config_gosai.yaml")
    final_model = diffusion_gosai.Diffusion(config=cfg).to(device)
    if best_model_path is not None:
        final_model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("testing final performances on SVDD way")
    final_sample_new = generate_xt_list(
            args=args,
            generative_model=final_model,
            device=device,
            timesteps=timesteps,
            dt=dt,
            reward_model=eval_models,
            make_svdd=True,
    )[0]

    # Pred-Activity, rewards_eval is more important, higher better
    reward_argmax = eval_models.cal_pred_activity(final_sample_new, reward_mode="train")  # bs
    reward_argmax_eval = eval_models.cal_pred_activity(final_sample_new, reward_mode="eval")  # bs
    rewards = [reward_argmax.detach().cpu().numpy()]
    rewards_eval = [reward_argmax_eval.detach().cpu().numpy()]
    # ATAC-Acc, higher better
    atac_reward = eval_models.cal_atac_acc(final_sample_new)  # bs
    atac_acc_rewards = [atac_reward > 0.5]
    # k-mer
    k_mer_count_dict, k_mer_length = batch_agg_count_k_mers(
        dna_seqs=final_sample_new.detach().cpu().numpy(),
        counts={},
        length=0,
    )

    rewards_mid = np.median(np.concatenate(rewards)).item()
    rewards_mid_eval = np.median(np.concatenate(rewards_eval)).item()
    atac_acc_rewards_mean = np.mean(np.concatenate(atac_acc_rewards)).item()
    k_mer_value = eval_models.compare_kmer(
        kmer2=k_mer_count_dict,
        n_sp2=k_mer_length,
    ).item()
    rewards_mid_std = np.std(np.concatenate(rewards)).item()
    rewards_eval_std = np.std(np.concatenate(rewards_eval)).item()
    atac_acc_rewards_std = np.std(np.concatenate(atac_acc_rewards)).item()

    return {
        "final_SVDD_median_reward": rewards_mid,
        "final_SVDD_median_reward_eval": rewards_mid_eval,
        "final_SVDD_mean_atac_acc": atac_acc_rewards_mean,
        "final_SVDD_k_mer": k_mer_value,
        "final_SVDD_std_reward": rewards_mid_std,
        "final_SVDD_std_reward_eval": rewards_eval_std,
        "final_SVDD_std_atac_acc": atac_acc_rewards_std,
    }


def best_of_n_test(
        args,
        eval_models,
        timesteps, dt, device, best_model_path=None, pre_model=None,
):
    GlobalHydra.instance().clear()
    # Initialize Hydra and compose the configuration
    initialize(config_path="configs_gosai", job_name="load_model")
    cfg = compose(config_name="config_gosai.yaml")
    final_model = diffusion_gosai.Diffusion(config=cfg).to(device)
    if best_model_path is not None:
        final_model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("testing final performances on best of N")
    final_sample_new = generate_xt_list(
        args=args,
        generative_model=final_model,
        device=device,
        timesteps=timesteps,
        dt=dt,
        reward_model=eval_models,
        num_best_of_N=args.best_of_N,
    )[0]

    # Pred-Activity, rewards_eval is more important, higher better
    reward_argmax = eval_models.cal_pred_activity(final_sample_new, reward_mode="train")  # bs
    reward_argmax_eval = eval_models.cal_pred_activity(final_sample_new, reward_mode="eval")  # bs
    rewards = [reward_argmax.detach().cpu().numpy()]
    rewards_eval = [reward_argmax_eval.detach().cpu().numpy()]
    # ATAC-Acc, higher better
    atac_reward = eval_models.cal_atac_acc(final_sample_new)  # bs
    atac_acc_rewards = [atac_reward > 0.5]
    # k-mer
    k_mer_count_dict, k_mer_length = batch_agg_count_k_mers(
        dna_seqs=final_sample_new.detach().cpu().numpy(),
        counts={},
        length=0,
    )
    # log likelihood
    model_logl = pre_model.get_likelihood(
        x0=final_sample_new,
        num_steps=args.total_num_steps,
        eps=args.eps,
        n_samples=1,
    )  # bs
    model_logl_list = [model_logl.detach().cpu().numpy()]

    rewards_mid = np.median(np.concatenate(rewards)).item()
    rewards_mid_eval = np.median(np.concatenate(rewards_eval)).item()
    atac_acc_rewards_mean = np.mean(np.concatenate(atac_acc_rewards)).item()
    k_mer_value = eval_models.compare_kmer(
        kmer2=k_mer_count_dict,
        n_sp2=k_mer_length,
    ).item()
    rewards_mid_std = np.std(np.concatenate(rewards)).item()
    rewards_eval_std = np.std(np.concatenate(rewards_eval)).item()
    atac_acc_rewards_std = np.std(np.concatenate(atac_acc_rewards)).item()
    logl_mid = np.median(np.concatenate(model_logl_list)).item()
    logl_std = np.std(np.concatenate(model_logl_list)).item()

    return {
        "final_best_of_N_median_reward": rewards_mid,
        "final_best_of_N_median_reward_eval": rewards_mid_eval,
        "final_best_of_N_mean_atac_acc": atac_acc_rewards_mean,
        "final_best_of_N_k_mer": k_mer_value,
        "final_best_of_N_std_reward": rewards_mid_std,
        "final_best_of_N_std_reward_eval": rewards_eval_std,
        "final_best_of_N_std_atac_acc": atac_acc_rewards_std,
        "final_best_of_N_median_logl": logl_mid,
        "final_best_of_N_std_logl": logl_std,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_alg', type=str, default="sampling", choices=['SVDDtw', 'sampling'])
    parser.add_argument('--sample_M', type=int, default=10)
    parser.add_argument('--best_of_N', type=int, default=1)

    parser.add_argument('--gkd_lmbda', type=float, default=0.5, help="0 for pretrain roll in, 1 for student roll in")

    # distill loss params
    parser.add_argument('--teacher_alpha', type=float, default=1.0)
    parser.add_argument('--reward_norm', type=str, default='none')

    # loss function
    parser.add_argument('--loss_func', type=str, default="KL")

    # DDPO
    parser.add_argument("--ratio_clip", type=float, default=1e-4)
    parser.add_argument("--adv_norm", action='store_true')

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--total_num_steps', type=int, default=128)
    parser.add_argument('--num_accum_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--student_initialize_pretrain', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--wandb_name', type=str, default="debug", help="name for wandb run", required=False)
    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--wandb_group', type=str, default="")
    parser.add_argument('--data_base_path', default="/data484_2/xsu2/PolicyDistillation/DNA", type=str)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--target_update_interval', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_every_n_epochs', type=int, default=250)

    # task
    parser.add_argument('--task', type=str, default="dna", choices=['dna'])
    parser.add_argument('--reward', type=str, default="pred_activity")
    parser.add_argument('--svdd_baseline', action='store_true')
    parser.add_argument('--best_of_n_baseline', action='store_true')

    # use pretrained roll-in
    parser.add_argument("--use_decoding", action='store_true')
    parser.add_argument('--decoding_save_path', type=str, default='/data484_2/xsu2/PolicyDistillation/DNA/dna_pred_activity_bestofN_32/samples/all_samples.fasta')
    parser.add_argument('--use_schedule_rollin', action='store_true')

    parser.add_argument('--test_model_path', type=str, default="")
    parser.add_argument('--only_test', action='store_true')

    args = parser.parse_args()

    run(args)
