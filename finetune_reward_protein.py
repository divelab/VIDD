import argparse
import wandb
import numpy as np
import torch
import random
from datetime import datetime
import torch.nn.functional as F
import os, math, copy
from tqdm import tqdm
# from sequence_models.datasets import UniRefDataset

from evaluations.eval_models import initialize_eval_model
from losses.rl_loss import *
from models.gen_models import initialize_gen_model, generate_xt_list
from models.protein_gen_models import ProteinGenDiffusion
from evaluations.protein_utils import batch_tokenize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if 'iptm' in args.reward:
        result_save_folder = os.path.join('output', f"{args.task}_{args.bind_target}_{args.reward}_{args.wandb_name}")
    else:
        result_save_folder = os.path.join('output', f"{args.task}_{args.reward}_{args.wandb_name}")
    save_folders_model = os.path.join(result_save_folder, "models")
    os.makedirs(save_folders_model, exist_ok=True)
    args.result_save_folder = result_save_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """initialize diffusion model & reward model"""
    model_collections = initialize_gen_model(args, device)
    pre_model, old_model, new_model = model_collections['pre_model'], model_collections['old_model'], model_collections['new_model']
    eval_models = initialize_eval_model(args=args, device=device, result_save_folder=result_save_folder)

    if args.only_test:
        with torch.no_grad():
            result_dict = best_of_n_test(args=args, eval_models=eval_models, device=device,
                                         num_best_of_N=args.best_of_N, best_model_path=args.test_model_path)
            print(result_dict)
            # plddt_scores = [item[0] * 100 for item in final_each_reward_list[:32]]
            # sum(score > 60 for score in plddt_scores) / len(plddt_scores) * 100
            # sum(score > 70 for score in plddt_scores) / len(plddt_scores) * 100
            # sum(score > 80 for score in plddt_scores) / len(plddt_scores) * 100

            # sample_new = generate_xt_list(
            #         args=args,
            #         generative_model=new_model,
            #         device=device,
            #         seq_len=args.gen_len,
            #         unmask_K=args.unmask_K,
            #         reward_model=eval_models,
            #         num_best_of_N=8,
            #         train_stage=True,
            #     )[0]
            # cur_reward_list, cur_each_reward_list = eval_models.reward_metrics(S_sp=sample_new, save_pdb=False, return_all_reward_term=True)  # [bs]
            # print(np.mean(cur_reward_list))
        return

    if args.svdd_baseline:
        svdd_result_dict = svdd_test(args=args, eval_models=eval_models, device=device)
        wandb.log(svdd_result_dict)
        wandb.finish()
        return
    if args.best_of_n_baseline:
        result_dict = best_of_n_test(args=args, eval_models=eval_models, device=device, num_best_of_N=args.best_of_N)
        wandb.log(result_dict)
        wandb.finish()
        return

    # prepare dataset
    new_model.train()
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    if args.resume_train:
        checkpoint = torch.load(args.resume_train_path)
        new_model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

    # # initialize dataset length
    # reference_dataset = UniRefDataset(args.uniref_path,'rtest', structure=False, max_len=args.gen_len)
    # len_refer_dataset = len(reference_dataset)

    best_plddt_reward = float('-inf')
    best_rewards_eval, best_reward_std, best_diversity = float('-inf'), float('-inf'), float('-inf')
    best_model_path = os.path.join(save_folders_model, f'model_best.ckpt')
    for epoch_num in tqdm(range(args.num_epochs), desc="training epochs"):
        rewards, rewards_each_term = [], []
        diversity = []
        losses = []
        all_reward_weights, all_log_probs = [], []
        tot_grad_norm = 0.0
        prev_new_model = copy.deepcopy(new_model.state_dict())
        for _step in range(args.num_accum_steps):
            cur_len = args.gen_len
            unmask_K = args.unmask_K
            total_num_steps = range(args.total_num_steps)

            with torch.no_grad():
                """generate dataset"""
                sample_new, last_x_list_new, condt_list_new, conds_list_new, \
                    move_chance_t_list_new, move_chance_s_list_new, copy_flag_list_new = \
                    generate_xt_list(
                        args=args,
                        generative_model=new_model,
                        device=device,
                        seq_len=cur_len,
                        unmask_K=unmask_K,
                        reward_model=eval_models,
                        num_best_of_N=args.best_of_N,
                        train_stage=True,
                    )

                gkd_lmbda = args.gkd_lmbda

                if random.random() < gkd_lmbda:
                    # student model as roll-in distribution
                    if args.old_roll_in:
                        sample_old, last_x_list_old, condt_list_old, conds_list_old, \
                            move_chance_t_list_old, move_chance_s_list_old, copy_flag_list_old = \
                            generate_xt_list(
                                args=args,
                                generative_model=old_model,
                                device=device,
                                seq_len=cur_len,
                                unmask_K=unmask_K,
                                reward_model=eval_models,
                                num_best_of_N=args.best_of_N,
                                train_stage=True,
                            )
                        last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                            last_x_list_old, condt_list_old, conds_list_old, move_chance_t_list_old, move_chance_s_list_old, copy_flag_list_old
                        current_roll_in = "old"
                        final_sample = sample_old
                    else:
                        last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                            last_x_list_new, condt_list_new, conds_list_new, move_chance_t_list_new, move_chance_s_list_new, copy_flag_list_new
                        final_sample = sample_new
                        current_roll_in = "student"
                else:
                    sample_pre, last_x_list_pre, condt_list_pre, conds_list_pre, \
                        move_chance_t_list_pre, move_chance_s_list_pre, copy_flag_list_pre = \
                        generate_xt_list(
                            args=args,
                            generative_model=pre_model,
                            device=device,
                            seq_len=cur_len,
                            unmask_K=unmask_K,
                            reward_model=eval_models,
                            num_best_of_N=args.best_of_N,
                            train_stage=True,
                        )
                    # classifier free fine-tuning
                    last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                        last_x_list_pre, condt_list_pre, conds_list_pre, move_chance_t_list_pre, move_chance_s_list_pre, copy_flag_list_pre
                    final_sample = sample_pre
                    current_roll_in = "pretrain"

                # shuffle list
                combined = list(zip(
                    last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list
                ))
                random.shuffle(combined)
                last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = zip(*combined)
                last_x_list = list(last_x_list)
                condt_list = list(condt_list)
                conds_list = list(conds_list)
                move_chance_t_list = list(move_chance_t_list)
                move_chance_s_list = list(move_chance_s_list)
                copy_flag_list = list(copy_flag_list)

            """observe performances of student generated samples"""
            cur_reward_list, cur_each_reward_list = eval_models.reward_metrics(S_sp=sample_new, save_pdb=False, return_all_reward_term=True)  # [bs]
            cur_diversity = eval_models.calc_diversity(sample_new)  # float number
            rewards.extend(cur_reward_list)
            rewards_each_term.extend(cur_each_reward_list)
            diversity.append(cur_diversity)

            """model training"""
            new_model.train()
            total_loss, total_reward_weight, total_log_probs = [], [], []
            for random_t in tqdm(total_num_steps, disable=True if args.total_num_steps == 1 else False, desc="timestep iteration"):
                xt = last_x_list[random_t]
                # xt_seq = xt.argmax(dim=-1)
                condt = condt_list[random_t]
                # conds = conds_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                # move_chance_s = move_chance_s_list[random_t]
                # copy_flag = copy_flag_list[random_t]  # [bsz, seqlen, 1], x_t is not mask: 1, x_t is mask: 0

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    reward_weight, log_probs, loss = protein_loss(
                        args=args,
                        new_model=new_model,
                        old_model=old_model,
                        pre_model=pre_model,
                        eval_models=eval_models,
                        xt=xt,
                        condt=condt,
                        move_chance_t=move_chance_t,
                        unmask_K=unmask_K,
                        final_sample=final_sample,
                        current_roll_in=current_roll_in,
                    )

                if loss is not None:
                    total_loss.append(loss)
                if reward_weight is not None:
                    total_reward_weight.append(reward_weight)
                if log_probs is not None:
                    total_log_probs.append(log_probs)

            loss = torch.stack(total_loss).mean()
            reward_weight = torch.stack(total_reward_weight).mean()
            log_probs = torch.stack(total_log_probs).mean()

            loss = loss / args.num_accum_steps
            reward_weight = reward_weight / args.num_accum_steps
            log_probs = log_probs / args.num_accum_steps

            # loss.backward()
            scaler.scale(loss).backward()
            if (_step + 1) % args.num_accum_steps == 0:  # Gradient accumulation
                scaler.unscale_(optim)
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.grad_clip)
                tot_grad_norm += norm
                # optim.step()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            all_reward_weights.append(reward_weight.cpu().detach().numpy() * args.num_accum_steps)
            all_log_probs.append(log_probs.cpu().detach().numpy() * args.num_accum_steps)

        if args.target_update_interval > 0 and epoch_num % args.target_update_interval == 0 and epoch_num != 0:
            # update theta_old by theta
            if args.loss_func == 'PPO' or args.loss_func == 'DDPO':
                old_model.load_state_dict(prev_new_model)
            else:
                old_model.load_state_dict(new_model.state_dict())

        """record performances"""
        result_dict = {}
        losses = np.array(losses)
        reward_weights = np.array(all_reward_weights)
        log_probs = np.array(all_log_probs)

        rewards_eval = np.mean(rewards).item()
        rewards_std = np.std(rewards).item()
        diversity_mean = np.mean(diversity).item()
        diversity_std = np.std(diversity).item()
        cur_result_dict = {
            "mean_reward": rewards_eval,
            "std_reward": rewards_std,
            "mean_diversity": diversity_mean,
            "std_diversity": diversity_std,
        }

        rewards_each_term = np.array(rewards_each_term)  # bs * num reward
        rewards_each_term_mean = np.mean(rewards_each_term, axis=0)
        rewards_each_term_std = np.std(rewards_each_term, axis=0)
        # record each reward term
        for r_idx, each_reward_name in enumerate(args.reward.split(",")):
            cur_result_dict[f"{each_reward_name}_mean_reward"] = rewards_each_term_mean[r_idx]
            cur_result_dict[f"{each_reward_name}_std_reward"] = rewards_each_term_std[r_idx]
            if each_reward_name == 'plddt':
                plddt_reward = rewards_each_term_mean[r_idx]

        if "plddt" in args.reward.split(","):
            plddt_idx = args.reward.split(",").index('plddt')
            plddt_scores = rewards_each_term[:,plddt_idx]
            plddt70 = sum(score > 0.7 for score in plddt_scores) / len(plddt_scores) * 100
            # print(f"plddt > 70% is {plddt70}")
            cur_result_dict[f'plddt_greater_70'] = plddt70

        loss_dict = {}
        if len(losses) > 0:
            loss_dict['mean_loss'] = np.mean(losses)
        if len(reward_weights) > 0:
            loss_dict['mean_reward_weights'] = np.mean(reward_weights)
        if len(log_probs) > 0:
            loss_dict['mean_log_probs'] = np.mean(log_probs)

        result_dict.update(cur_result_dict)
        result_dict.update(loss_dict)
        # print(result_dict)
        wandb.log(result_dict, step=epoch_num)

        # save model
        checkpoint = {
            'model': new_model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch_num,
            'scaler': scaler.state_dict() if args.use_amp else None,
            'args': args,
            'best_reward': best_rewards_eval,
        }

        # save at certain intervals
        if (epoch_num + 1) % args.save_every_n_epochs == 0 and args.wandb_name != 'debug':
            model_path = os.path.join(save_folders_model, f'model_{epoch_num}.ckpt')
            torch.save(checkpoint, model_path)
            print(f"Model saved at epoch {epoch_num}")
        # save best model
        if rewards_eval > best_rewards_eval and args.wandb_name != 'debug':
            best_rewards_eval = rewards_eval
            best_reward_std = rewards_std
            best_diversity = diversity_mean
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved at epoch {epoch_num}")
        # save best reward or best plddt
        if plddt_reward > best_plddt_reward and args.wandb_name != 'debug':
            best_plddt_reward = plddt_reward
            torch.save(checkpoint, os.path.join(save_folders_model, f'model_best_plddt.ckpt'))
            print(f"Best PLDDT Model saved at epoch {epoch_num}")
        # save the last model
        last_model_path = os.path.join(save_folders_model, f'last.ckpt')
        torch.save(checkpoint, last_model_path)

    wandb.log({
        "final_mean_reward": best_rewards_eval,
        "final_std_reward": best_reward_std,
        "best_diversity": best_diversity,
    })

    wandb.finish()


def svdd_test(
        eval_models,
        args, device, best_model_path=None,
):
    final_model = ProteinGenDiffusion(args).to(device)
    final_model.eval()
    for param in final_model.parameters():
        param.requires_grad = False
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        final_model.load_state_dict(checkpoint['model'])

    final_sample_new = generate_xt_list(
            args=args,
            generative_model=final_model,
            device=device,
            seq_len=args.gen_len,
            unmask_K=args.unmask_K,
            reward_model=eval_models,
            make_svdd=True,
    )[0]
    final_reward_list = eval_models.reward_metrics(S_sp=final_sample_new, save_pdb=True, save_pdb_name="p")  # [bs]

    cur_diversity = eval_models.calc_diversity(final_sample_new)  # float number

    return {
        "final_mean_reward_SVDD": np.mean(final_reward_list).item(),
        "final_std_reward_SVDD": np.std(final_reward_list).item(),
        "final_diversity_SVDD": cur_diversity,
    }


def best_of_n_test(
        eval_models,
        args, device, best_model_path=None,
        num_best_of_N=1,
):
    final_model = ProteinGenDiffusion(args).to(device)
    final_model.eval()
    for param in final_model.parameters():
        param.requires_grad = False
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        final_model.load_state_dict(checkpoint['model'])

    final_sample_new = generate_xt_list(
            args=args,
            generative_model=final_model,
            device=device,
            seq_len=args.gen_len,
            unmask_K=args.unmask_K,
            reward_model=eval_models,
            num_best_of_N=num_best_of_N,
    )[0]
    final_reward_list, final_each_reward_list = eval_models.reward_metrics(S_sp=final_sample_new, save_pdb=True, save_pdb_name=f"best_of_{args.best_of_N}", return_all_reward_term=True)  # [bs]

    cur_diversity = eval_models.calc_diversity(final_sample_new)  # float number

    if 'plddt' in args.reward.split(","):
        plddt_idx = args.reward.split(",").index('plddt')
        plddt_scores = [item[plddt_idx] * 100 for item in final_each_reward_list[:32]]
        plddt70 = sum(score > 70 for score in plddt_scores) / len(plddt_scores) * 100
        print(f"plddt > 70% is {plddt70}")

    result_dict = {
        f"final_mean_reward_best_of_N": np.mean(final_reward_list).item(),
        f"final_std_reward_best_of_N": np.std(final_reward_list).item(),
        f"final_diversity_best_of_N": cur_diversity,
    }

    final_each_reward_list = np.array(final_each_reward_list)  # bs * num reward
    final_each_reward_list_mean = np.mean(final_each_reward_list, axis=0)
    final_each_reward_list_std = np.std(final_each_reward_list, axis=0)
    # record each reward term
    for r_idx, each_reward_name in enumerate(args.reward.split(",")):
        result_dict[f"{each_reward_name}_mean_reward"] = final_each_reward_list_mean[r_idx]
        result_dict[f"{each_reward_name}_std_reward"] = final_each_reward_list_std[r_idx]

    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_alg', type=str, default="sampling", choices=['SVDDtw', 'sampling'])
    parser.add_argument('--SVDD_num_candidate', type=int, default=20)
    parser.add_argument('--best_of_N', type=int, default=1)
    parser.add_argument('--gkd_lmbda', type=float, default=0.5, help="0 for pretrain roll in, 1 for student roll in")

    # distill loss params
    parser.add_argument('--teacher_alpha', type=float, default=1.0)
    parser.add_argument('--reward_norm', type=str, default='none')
    parser.add_argument('--logits_alpha', type=float, default=1.0)

    # loss function
    parser.add_argument('--loss_func', type=str, default="KL")

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--total_num_steps', type=int, default=1)
    parser.add_argument('--num_accum_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--student_initialize_pretrain', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--wandb_name', type=str, default="debug", help="name for wandb run", required=False)
    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--wandb_group', type=str, default="")
    parser.add_argument('--data_base_path', default="", type=str)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--target_update_interval', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_every_n_epochs', type=int, default=50)

    # task
    parser.add_argument('--task', type=str, default="protein", choices=['protein'])
    parser.add_argument('--reward', type=str, default="")
    parser.add_argument('--reward_weight', type=str, default='1')

    # protein task params
    parser.add_argument('--define_ss', type=str, default="b", choices=['a', 'b'])
    parser.add_argument('--bind_target', type=str, default="PDL1")
    parser.add_argument('--bind_target_folder', type=str, default="./target_proteins")
    parser.add_argument('--gen_len', type=int, default=256)
    parser.add_argument('--unmask_K', default=4, type=int)
    parser.add_argument('--folding_model', default="3b", choices=['650m', '3b'], type=str)

    # model testing
    parser.add_argument('--test_model_path', type=str, default="")
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--svdd_baseline', action='store_true')
    parser.add_argument('--best_of_n_baseline', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--resume_train_path', type=str, default="")

    parser.add_argument('--rs_gen_model', default='pre')
    parser.add_argument('--old_roll_in', action='store_true')

    # DDPO
    parser.add_argument("--ratio_clip", type=float, default=1e-4)
    parser.add_argument("--adv_norm", action='store_true')

    # forward KL implementation detail
    parser.add_argument('--reward_step', action='store_true')
    parser.add_argument('--reward_temp', action='store_true')
    parser.add_argument('--reward_estimate_times', default=1, type=int)
    parser.add_argument('--use_value_xt', action='store_true')
    parser.add_argument('--reward_clamp', default=1e6, type=float)

    # protein binding affinity
    parser.add_argument('--msa_mode', default="none", type=str, choices=['none', 'full_msa'])

    args = parser.parse_args()
    print(args)

    run(args)
