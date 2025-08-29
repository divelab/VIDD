import torch
import torch.nn.functional as F
import numpy as np


def calc_reward_dna(
        reward_model,
        input_x,
        input_t,
        # log_p_x0,
        gen_model,
):
    """
    first get x0, then evaluate
    """
    log_p_x0 = gen_model.forward(input_x, input_t)[:, :, :-1]
    x0_new = (log_p_x0.exp()).argmax(dim=-1)  # .exp() makes it 4-hot prob, sum=1 (bs*seq*4). x0_pre: bs*seq
    x0_new_one_hot = 1.0 * F.one_hot(x0_new, num_classes=4)
    sample_reward_x0_new = torch.transpose(x0_new_one_hot, 1, 2)
    preds = reward_model(sample_reward_x0_new).squeeze(-1)  # [bs, 3]
    reward = preds[:, 0]  # should maximize, shape: bs
    return reward


def temp_sampling(prediction_logits, temp=1.0):
    logits = prediction_logits / temp  # bs * len* dim
    probs = torch.softmax(logits, dim=-1)

    # Sample from categorical distribution
    predicted_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[0],
                                                                                             probs.shape[1])
    return predicted_tokens


@torch.no_grad()
def calc_reward_protein(
        reward_model,
        input_x,
        timestep,
        gen_model,
        args=None,
        mask_token_id=28,
        estimate_one_time=False,
):
    """
    first get x0, then evaluate
    """
    reward_list = []
    if args.reward_step and args.reward_temp and estimate_one_time is False:
        estimate_time = args.reward_estimate_times
    else:
        estimate_time = 1
    for reward_idx in range(estimate_time):
        if args.reward_step:
            B = input_x.shape[0]
            cur_seq = input_x.clone()
            masked_token = input_x == mask_token_id

            while masked_token.any():
                prediction_logits = gen_model(cur_seq, timestep)  # [B, L, V]
                if args.reward_temp:
                    predicted_tokens = temp_sampling(prediction_logits[:, :, 0:20])
                else:
                    prediction_probs = prediction_logits[:, :, 0:20].softmax(dim=-1)
                    predicted_tokens = prediction_probs.argmax(dim=-1)  # greedy decode: [B, L]

                for b in range(B):
                    mask_pos = torch.nonzero(masked_token[b], as_tuple=False).squeeze(1)  # shape: [num_masked_b]
                    if len(mask_pos) == 0:
                        continue

                    num_unmask = min(args.unmask_K, len(mask_pos))
                    selected = mask_pos[torch.randperm(len(mask_pos))[:num_unmask]]

                    cur_seq[b, selected] = predicted_tokens[b, selected]

                masked_token = (cur_seq == mask_token_id)

        else:
            prediction = gen_model(input_x, timestep)
            cur_seq = input_x * (input_x != 28) + torch.argmax(prediction[:, :, 0:20], dim=2) * (input_x == 28)

        reward = reward_model(S_sp=cur_seq, save_pdb=False)
        reward_list.append(reward)
    reward = np.array(reward_list).mean(axis=0).tolist()
    return reward


def dna_loss(
        args,
        new_model,
        old_model,
        pre_model,
        eval_models,
        xt,
        xt_seq,
        condt,
        conds,
        move_chance_t,
        move_chance_s,
        # copy_flag,
        current_roll_in,
        value_net=None,
        progress_t=-1,
        final_reward=None,
        sample_new=None,
        log_fs=None,
        xs_define=None,
):
    def gen_samples(gen_model):
        log_p_x0 = gen_model.forward(xt, condt)
        p_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)  # p xt-1, student model
        p_xs[:, :, new_model.mask_index] = move_chance_s[:, :, 0]

        copy_flag = (xt_seq != new_model.mask_index).to(xt_seq.dtype)
        batch_size, seq_len, vocab_size = p_xs.shape

        probs_2d = p_xs.reshape(-1, vocab_size)
        samples = torch.multinomial(probs_2d, num_samples=1)
        indices = samples.reshape(batch_size, seq_len)

        xs_seq = copy_flag * xt_seq + (1 - copy_flag) * indices
        return log_p_x0, p_xs, indices, xs_seq, vocab_size

    log_p_x0, p_xs, indices, xs_seq, vocab_size = gen_samples(new_model)
    log_p_x0_old, p_xs_old, indices_old, xs_old_seq, vocab_size = gen_samples(old_model)

    copy_flag = (xt_seq != new_model.mask_index).to(xt_seq.dtype)
    if args.rs_gen_model == 'pre':
        gen_model = pre_model
    elif args.rs_gen_model == 'new':
        gen_model = new_model
    elif args.rs_gen_model == 'old':
        gen_model = old_model
    else:
        raise NotImplementedError()
    # log_p_x0 = new_model.forward(xt, condt)  # [batch_size, sequence_length, 5]
    # log_p_x0_old = old_model.forward(xt, condt)
    # # log_p_x0_pre = pre_model.forward(last_x, condt)[:, :, :-1]
    # # if args.use_center_log_prob:
    # #     # TODO: change of log prob cannot be done here, maybe later?
    # #     log_p_x0 = log_p_x0 - log_p_x0.mean(dim=-1, keepdim=True)
    # #     log_p_x0_old = log_p_x0_old - log_p_x0_old.mean(dim=-1, keepdim=True)
    #
    # # compute loss
    # # xt -> p xt-1
    # p_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)  # p xt-1, student model
    # p_xs[:, :, new_model.mask_index] = move_chance_s[:, :, 0]
    # p_xs_old = log_p_x0_old.exp() * (move_chance_t - move_chance_s)  # p xt-1, old model
    # p_xs_old[:, :, old_model.mask_index] = move_chance_s[:, :, 0]
    #
    # # p xt-1 -> xt-1, temperature sampling, need randomness
    # copy_flag = (xt_seq != new_model.mask_index).to(xt_seq.dtype)
    # # xs = copy_flag * xt + (1 - copy_flag) * _sample_categorical(p_xs)
    # # indices_old = p_xs_old.argmax(dim=-1)
    #
    # batch_size, seq_len, vocab_size = p_xs_old.shape
    #
    # probs_2d = p_xs.reshape(-1, vocab_size)
    # samples = torch.multinomial(probs_2d, num_samples=1)
    # indices = samples.reshape(batch_size, seq_len)
    #
    # probs_2d = p_xs_old.reshape(-1, vocab_size)
    # samples = torch.multinomial(probs_2d, num_samples=1)
    # indices_old = samples.reshape(batch_size, seq_len)
    #
    # xs_seq = copy_flag * xt_seq + (1 - copy_flag) * indices
    # xs_old_seq = copy_flag * xt_seq + (1 - copy_flag) * indices_old

    # teacher distribution xt-1, based on old model
    v_xs = calc_reward_dna(
        reward_model=eval_models.reward_train,
        input_x=F.one_hot(xs_old_seq, num_classes=vocab_size).float(),
        input_t=conds,
        gen_model=gen_model,
    )
    if args.reward_norm == 'none':
        pass
    elif args.reward_norm == 'pos':
        v_xs = v_xs - v_xs.min().detach()
    elif args.reward_norm == 'normal':
        # v_xs = v_xs - v_xs.mean().detach()
        # v_xs = v_xs / (v_xs.abs().max().detach() + 1e-8)
        v_xs = (v_xs - v_xs.mean().detach()) / (v_xs.std().detach() + 1e-8)
    else:
        raise NotImplementedError()

    p_xs_select = torch.gather(p_xs.log(), dim=-1, index=xs_old_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

    gen_flag = (copy_flag == 0)  # [B, L]
    not_mask = (indices_old != new_model.mask_index)  # [B, L]
    valid_mask = gen_flag & not_mask  # [B, L]

    # valid_mask = (xs_old_seq != new_model.mask_index)
    p_xs_select_valid = p_xs_select * valid_mask  # [B, L]
    p_xs_select_valid_sum = p_xs_select_valid.sum(dim=1)  # [B]

    if args.loss_func == 'KL':
        v_xs = torch.exp(v_xs / args.teacher_alpha).detach()  # [bs]

        if args.use_value_xt:
            v_xt = calc_reward_dna(
                reward_model=eval_models.reward_train,
                input_x=xt.float(),
                input_t=condt,
                gen_model=gen_model,
            )
            if args.reward_norm == 'normal':
                v_xt = (v_xt - v_xt.mean().detach()) / (v_xt.std().detach() + 1e-8)
            v_xt = torch.exp(v_xt / args.teacher_alpha).detach()  # [bs]
            v_xs = v_xs / v_xt

        loss = -(v_xs * p_xs_select_valid_sum).mean()  # scalar

        reward_weight, log_probs = v_xs.mean(), p_xs_select_valid_sum.mean()

    elif args.loss_func == 'RW_MLE':
        x0_seq = sample_new
        v_x0 = eval_models.cal_pred_activity(x0_seq, reward_mode="train")
        if args.reward_norm == 'normal':
            # v_xs = v_xs - v_xs.mean().detach()
            # v_xs = v_xs / (v_xs.abs().max().detach() + 1e-8)
            v_x0 = (v_x0 - v_x0.mean().detach()) / (v_x0.std().detach() + 1e-8)

        teacher_weight = torch.exp(v_x0 / args.teacher_alpha).detach()  # [bs]

        p_x0_select = torch.gather(log_p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        valid_mask = (copy_flag == 0) & (x0_seq != new_model.mask_index)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]

        loss = -(teacher_weight * p_x0_select_valid_mean).mean()  # scalar
        reward_weight, log_probs = teacher_weight.mean(), p_x0_select_valid_mean.mean()

    elif args.loss_func == 'DDPO':
        # point to x0, not xt-1
        x0_seq = copy_flag * xt_seq + (1 - copy_flag) * sample_new  # x0
        # x0_reward = calc_reward_dna(
        #     reward_model=eval_models.reward_train,
        #     input_x=F.one_hot(x0_seq, num_classes=vocab_size).float(),
        #     input_t=conds,
        #     gen_model=pre_model,
        # )
        x0_reward = eval_models.cal_pred_activity(x0_seq, reward_mode="train")
        assert new_model.mask_index not in x0_seq
        assert torch.equal(x0_seq, sample_new)

        p_x0_select = torch.gather(log_p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_x0_select_old = torch.gather(log_p_x0_old, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

        valid_mask = (copy_flag == 0) & (sample_new != new_model.mask_index)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]
        p_x0_select_valid_old = p_x0_select_old * valid_mask  # [B, L]
        p_x0_select_valid_mean_old = p_x0_select_valid_old.mean(dim=1)  # [B]

        # log prob ratio
        unclamp_ratio = torch.exp(p_x0_select_valid_mean - p_x0_select_valid_mean_old.detach())
        ratio = torch.clamp(unclamp_ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)

        # advantage
        if args.adv_norm:
            adv = (x0_reward - x0_reward.mean()) / (x0_reward.std() + 1e-8)
        else:
            adv = x0_reward / args.teacher_alpha
        adv = adv.detach()

        loss = torch.max(-adv * unclamp_ratio, -adv * ratio)
        loss = loss.mean()
        reward_weight = -adv.mean()
        log_probs = ratio.float().mean()

    elif args.loss_func == 'DDPP':
        # log_r0 = final_reward
        log_xt = calc_reward_dna(
            reward_model=eval_models.reward_train,
            input_x=xt.float(),
            input_t=condt,
            gen_model=pre_model,
        )

        x0_seq = copy_flag * xt_seq + (1 - copy_flag) * sample_new  # x0
        assert new_model.mask_index not in x0_seq
        assert torch.equal(x0_seq, sample_new)
        # log_r0 = calc_reward_dna(
        #     reward_model=eval_models.reward_train,
        #     input_x=F.one_hot(x0_seq, num_classes=vocab_size).float(),
        #     input_t=conds,
        #     gen_model=pre_model,
        # )
        log_r0 = eval_models.cal_pred_activity(x0_seq, reward_mode="train")

        log_p_x0_pre = pre_model.forward(xt, condt)

        p_x0_select = torch.gather(log_p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_x0_select_pre = torch.gather(log_p_x0_pre, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

        valid_mask = (copy_flag == 0) & (sample_new != new_model.mask_index)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]
        p_x0_select_valid_pre = p_x0_select_pre * valid_mask  # [B, L]
        p_x0_select_valid_mean_pre = p_x0_select_valid_pre.mean(dim=1)  # [B]

        term = (p_x0_select_valid_mean - p_x0_select_valid_mean_pre.detach() -
                (log_r0 / args.teacher_alpha).detach() + (log_xt / args.teacher_alpha).detach())
        loss = (term ** 2).mean()
        reward_weight = (log_r0 / args.teacher_alpha).detach().mean()
        log_probs = p_x0_select_valid_mean.mean()

    else:
        raise NotImplementedError()

    return {
        'loss': loss,
        'reward_weight': reward_weight,
        'log_probs': log_probs,
    }


def protein_loss(
        args,
        new_model,
        old_model,
        pre_model,
        eval_models,
        xt,
        condt,
        move_chance_t,
        unmask_K=1,
        final_sample=None,
        current_roll_in=None,
):
    # if args.denoise_step_svdd:
    #     p_x0, xs, loc_set_fake, copy_flag = new_model.xt_to_xs_svdd(
    #         sample=xt, timestep=condt, loc_set=move_chance_t, reward_model=eval_models,
    #         unmask_K=unmask_K, num_candidate=args.SVDD_num_candidate)
    #     p_x0_old, xs_old, loc_set_fake, copy_flag = old_model.xt_to_xs_svdd(
    #         sample=xt, timestep=condt, loc_set=move_chance_t, reward_model=eval_models,
    #         unmask_K=unmask_K, num_candidate=args.SVDD_num_candidate)
    # else:
    p_x0, xs, loc_set_fake, copy_flag = new_model.xt_to_xs(xt, condt, move_chance_t, unmask_K=unmask_K)
    pred_mean = None

    if args.rs_gen_model == 'pre':
        gen_model_v_xs = pre_model.model
    elif args.rs_gen_model == 'new':
        gen_model_v_xs = new_model.model
    elif args.rs_gen_model == 'old':
        gen_model_v_xs = old_model.model
    elif args.rs_gen_model == 'adapt':
        if current_roll_in == 'pretrain':
            gen_model_v_xs = pre_model.model
        elif current_roll_in == 'old':
            gen_model_v_xs = old_model.model
        elif current_roll_in == 'student':
            gen_model_v_xs = new_model.model
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    if args.loss_func == 'KL':
        p_x0_old, xs_old, loc_set_fake, copy_flag = old_model.xt_to_xs(xt, condt, move_chance_t, unmask_K=unmask_K)

        # value at xt-1
        v_xs = calc_reward_protein(eval_models.reward_metrics, xs_old, condt, gen_model=gen_model_v_xs,
                                   args=args)  # [bs]
        v_xs = torch.tensor(v_xs, device=p_x0.device)
        if args.reward_norm == 'normal':
            v_xs = (v_xs - v_xs.mean().detach()) / (v_xs.std().detach() + 1e-8)
        v_xs = torch.exp(v_xs / args.teacher_alpha).detach()  # [bs]

        # value at xt
        if args.use_value_xt:
            v_xt = calc_reward_protein(eval_models.reward_metrics, xt, condt, gen_model=gen_model_v_xs,
                                       args=args)  # [bs]
            v_xt = torch.tensor(v_xt, device=p_x0.device)
            if args.reward_norm == 'normal':
                v_xt = (v_xt - v_xt.mean().detach()) / (v_xt.std().detach() + 1e-8)
            v_xt = torch.exp(v_xt / args.teacher_alpha).detach()  # [bs]
            v_xs = v_xs / v_xt

        valid_mask = (copy_flag == 0) & (xs_old != new_model.tokenizer.mask_id)
        # select corresponding dimension of p_x0
        p_xs_select = torch.gather(p_x0, dim=-1, index=xs_old.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_xs_select_old = torch.gather(p_x0_old, dim=-1, index=xs_old.unsqueeze(-1)).squeeze(-1)  # [B, L]

        p_xs_select_valid = p_xs_select * valid_mask  # [B, L]
        p_xs_select_valid_sum = p_xs_select_valid.sum(dim=1)  # [B]
        p_xs_select_valid_old = p_xs_select_old * valid_mask  # [B, L]
        p_xs_select_valid_old_sum = p_xs_select_valid_old.sum(dim=1)  # [B]

        prob_term = -p_xs_select_valid_sum

        loss = (v_xs * prob_term).mean()  # scalar

        reward_weight, log_probs = v_xs.mean(), prob_term.mean()

    elif args.loss_func == 'RW_MLE':
        x0_seq = final_sample
        v_x0 = calc_reward_protein(eval_models.reward_metrics, x0_seq, condt, gen_model=pre_model.model, args=args, estimate_one_time=True)
        v_x0 = torch.tensor(v_x0, device=p_x0.device)
        if args.reward_norm == 'normal' and (not args.use_value_xt):
            v_x0 = (v_x0 - v_x0.mean().detach()) / (v_x0.std().detach() + 1e-8)

        v_x0 = torch.exp(v_x0 / args.teacher_alpha).detach()  # [bs]

        # calculate value at xt
        if args.use_value_xt:
            v_xt = calc_reward_protein(eval_models.reward_metrics, xt, condt, gen_model=gen_model_v_xs,
                                       args=args)  # [bs]
            v_xt = torch.tensor(v_xt, device=p_x0.device)
            v_xt = torch.exp(v_xt / args.teacher_alpha).detach()  # [bs]
            v_x0 = v_x0 / (v_xt + torch.finfo(v_xt.dtype).tiny)
            v_x0 = torch.clamp(v_x0, min=1 / args.reward_clamp, max=args.reward_clamp)
            if args.reward_norm == 'normal':
                v_x0 = (v_x0 - v_x0.mean().detach()) / (v_x0.std().detach() + 1e-8)

        p_x0_select = torch.gather(p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

        valid_mask = (copy_flag == 0) & (final_sample != new_model.tokenizer.mask_id)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]

        loss = -(v_x0 * p_x0_select_valid_mean).mean()  # scalar

        reward_weight, log_probs = v_x0.max(), p_x0_select_valid_mean.mean()

    elif args.loss_func == 'DDPO':
        p_x0_old, xs_old, loc_set_fake, copy_flag = old_model.xt_to_xs(xt, condt, move_chance_t, unmask_K=unmask_K)

        x0_seq = copy_flag * xt + (1 - copy_flag) * final_sample  # x0
        assert new_model.tokenizer.mask_id not in x0_seq
        assert torch.equal(x0_seq, final_sample)
        x0_reward = calc_reward_protein(eval_models.reward_metrics, x0_seq, condt, gen_model=pre_model.model, args=args)
        x0_reward = torch.tensor(x0_reward, device=p_x0.device)

        p_x0_select = torch.gather(p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_x0_select_old = torch.gather(p_x0_old, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

        valid_mask = (copy_flag == 0) & (final_sample != new_model.tokenizer.mask_id)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]
        p_x0_select_valid_old = p_x0_select_old * valid_mask  # [B, L]
        p_x0_select_valid_mean_old = p_x0_select_valid_old.mean(dim=1)  # [B]

        # log prob ratio
        unclamp_ratio = torch.exp(p_x0_select_valid_mean - p_x0_select_valid_mean_old.detach())
        ratio = torch.clamp(unclamp_ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)

        # advantage
        if args.adv_norm:
            adv = (x0_reward - x0_reward.mean()) / (x0_reward.std() + 1e-8)
        else:
            adv = x0_reward / args.teacher_alpha
        adv = adv.detach()

        loss = torch.max(-adv * unclamp_ratio, -adv * ratio)
        loss = loss.mean()
        reward_weight = -adv.mean()
        log_probs = ratio.float().mean()

    elif args.loss_func == 'DDPP':
        log_xt = calc_reward_protein(eval_models.reward_metrics, xt, condt, gen_model=pre_model.model, args=args)
        log_xt = torch.tensor(log_xt, device=p_x0.device)

        x0_seq = copy_flag * xt + (1 - copy_flag) * final_sample  # x0
        assert new_model.tokenizer.mask_id not in x0_seq
        assert torch.equal(x0_seq, final_sample)
        log_r0 = calc_reward_protein(eval_models.reward_metrics, x0_seq, condt, gen_model=pre_model.model, args=args)
        log_r0 = torch.tensor(log_r0, device=p_x0.device)

        p_x0_pre, xs_pre, loc_set_fake, copy_flag = pre_model.xt_to_xs(xt, condt, move_chance_t, unmask_K=unmask_K,
                                                                       input_log_prob_mean=pred_mean)

        p_x0_select = torch.gather(p_x0, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_x0_select_pre = torch.gather(p_x0_pre, dim=-1, index=x0_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]

        valid_mask = (copy_flag == 0) & (final_sample != new_model.tokenizer.mask_id)  # [B, L]

        p_x0_select_valid = p_x0_select * valid_mask  # [B, L]
        p_x0_select_valid_mean = p_x0_select_valid.mean(dim=1)  # [B]
        p_x0_select_valid_pre = p_x0_select_pre * valid_mask  # [B, L]
        p_x0_select_valid_mean_pre = p_x0_select_valid_pre.mean(dim=1)  # [B]

        v_x0_weight = (log_r0 / args.teacher_alpha).detach()
        v_xt_weight = (log_xt / args.teacher_alpha).detach()
        term = (p_x0_select_valid_mean - p_x0_select_valid_mean_pre.detach() - v_x0_weight + v_xt_weight)
        loss = (term ** 2).mean()

        reward_weight = (-v_x0_weight + v_xt_weight).mean()
        log_probs = (p_x0_select_valid_mean - p_x0_select_valid_mean_pre.detach()).mean()

    else:
        raise NotImplementedError()

    return reward_weight, log_probs, loss
