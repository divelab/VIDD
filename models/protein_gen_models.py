import copy

import numpy as np
import torch
import torch.nn as nn

from evodiff.pretrained import OA_DM_38M


class ProteinGenDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        checkpoint = OA_DM_38M()
        self.model, collater, self.tokenizer, scheme = checkpoint

    def xt_to_xs(
            self,
            sample,
            timestep,
            loc_set,
            unmask_K=1,
            return_log_prob_mean=False,
            input_log_prob_mean=None,
    ):
        unmask_K = min(unmask_K, len(loc_set[0]))
        prediction = self.model(sample, timestep)  # logit

        # select position
        # loc_list = np.random.randint(len(loc_set[0]), size=self.args.batch_size)  # Choose random location
        # pes_index = [list(loc_set[i])[loc_list[i]] for i in range(self.args.batch_size)]  # Which position

        pes_index = []
        for i in range(self.args.batch_size):
            loc_i = list(loc_set[i])
            chosen = list(np.random.choice(loc_i, size=unmask_K, replace=False))
            pes_index.append(chosen)

        p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
            self.args.batch_size)])  # sample at location i (random), dont let it predict non-standard AA
        p = torch.nn.functional.softmax(p, dim=-1)  # bs * K * 20
        p_sample = torch.multinomial(p.reshape(-1, p.shape[-1]), num_samples=1).reshape(self.args.batch_size, unmask_K, 1)
        # bs * K * 1. temperature sampling
        sample_fake = sample.clone()
        # for iii in range(self.args.batch_size):
        #     sample_fake[iii, pes_index[iii]] = p_sample.squeeze(1)[iii]
        for iii in range(self.args.batch_size):
            for k, pos in enumerate(pes_index[iii]):
                sample_fake[iii, pos] = p_sample.squeeze(1)[iii, k]

        # update sample
        loc_set_fake = copy.deepcopy(loc_set)
        # for jjj in range(self.args.batch_size):
        #     loc_set_fake[jjj].remove(pes_index[jjj])
        for i in range(self.args.batch_size):
            for pos in pes_index[i]:
                loc_set_fake[i].remove(pos)

        copy_flag = (sample != self.tokenizer.mask_id).to(sample.dtype)

        prediction = torch.log_softmax(prediction / self.args.logits_alpha, dim=-1)
        return prediction, sample_fake, loc_set_fake, copy_flag

    def xt_to_xs_svdd(self, sample, timestep, loc_set, reward_model, unmask_K=1, num_candidate=20):
        unmask_K = min(unmask_K, len(loc_set[0]))
        prediction = self.model(sample, timestep)  # p_0

        # select position
        # loc_list = np.random.randint(len(loc_set[0]), size=self.args.batch_size)  # Choose random location
        # pes_index = [list(loc_set[i])[loc_list[i]] for i in range(self.args.batch_size)]  # Which position

        next_candidate = []
        pes_index_list = []
        for candidate_idx in range(num_candidate):
            pes_index = []
            for i in range(self.args.batch_size):
                loc_i = list(loc_set[i])
                chosen = list(np.random.choice(loc_i, size=unmask_K, replace=False))
                pes_index.append(chosen)

            pes_index_list.append(pes_index)

            p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(self.args.batch_size)])  # sample at location i (random), dont let it predict non-standard AA
            p = torch.nn.functional.softmax(p, dim=-1)  # bs * K * 20
            p_sample = torch.multinomial(p.reshape(-1, p.shape[-1]), num_samples=1).reshape(self.args.batch_size, unmask_K, 1)
            # bs * K * 1. temperature sampling
            sample_fake = sample.clone()
            # for iii in range(self.args.batch_size):
            #     sample_fake[iii, pes_index[iii]] = p_sample.squeeze(1)[iii]
            for iii in range(self.args.batch_size):
                for k, pos in enumerate(pes_index[iii]):
                    sample_fake[iii, pos] = p_sample.squeeze(1)[iii, k]
            next_candidate.append(sample_fake.clone())

        # reward & selection
        reward_list = np.zeros((self.args.batch_size, num_candidate))
        for jjj in range(num_candidate):
            prediction_jjj = self.model(next_candidate[jjj], timestep)
            next_seq = (next_candidate[jjj] * (next_candidate[jjj] != 28) +
                        torch.argmax(prediction_jjj[:, :, 0:20], dim=2) * (next_candidate[jjj] == 28))
            reward_hoge = reward_model.reward_metrics(S_sp=next_seq, save_pdb=False)
            reward_list[:, jjj] = reward_hoge
        next_index = np.argmax(reward_list, 1)

        # update sample
        next_candidate = torch.stack(next_candidate)
        sample_fake = torch.stack([next_candidate[next_index[i], i, :] for i in range(self.args.batch_size)])
        loc_set_fake = copy.deepcopy(loc_set)
        for jjj in range(self.args.batch_size):
            for pos in pes_index_list[next_index[jjj]][jjj]:
                loc_set_fake[jjj].remove(pos)

        # loc_set_fake = copy.deepcopy(loc_set)
        # # for jjj in range(self.args.batch_size):
        # #     loc_set_fake[jjj].remove(pes_index[jjj])
        # for i in range(self.args.batch_size):
        #     for pos in pes_index[i]:
        #         loc_set_fake[i].remove(pos)

        copy_flag = (sample != self.tokenizer.mask_id).to(sample.dtype)

        prediction = torch.log_softmax(prediction / self.args.logits_alpha, dim=-1)
        return prediction, sample_fake, loc_set_fake, copy_flag

