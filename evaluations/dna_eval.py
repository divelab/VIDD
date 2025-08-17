import torch
import grelu
import pandas as pd
import os
from grelu.lightning import LightningModel
import grelu.data.preprocess
import grelu.data.dataset
import numpy as np
from typing import Callable, Union, List
from scipy.linalg import sqrtm
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from collections import Counter
from grelu.interpret.motifs import scan_sequences

from evaluations import dna_utils


class DNAEvalMetrics:
    def __init__(
            self,
            base_path,
            device,
    ):
        self.base_path = base_path
        self.device = device

        # Pred-Activity
        self.reward_train = self.initialize_reward_split_model(mode='train')
        self.reward_eval = self.initialize_reward_split_model(mode='eval')

        # ATAC-Acc
        self.atac_acc = LightningModel.load_from_checkpoint(
            os.path.join(base_path, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt'), map_location=device)
        self.atac_acc.eval()

        # kmer
        (self.highexp_kmers_99, self.n_highexp_kmers_99,
         self.highexp_kmers_999, self.n_highexp_kmers_999,
         self.highexp_set_sp_clss_999, self.highexp_preds_999,
         self.highexp_seqs_999) = self.cal_highexp_kmers(return_clss=True)

        # motif
        # motif_count_top = scan_sequences(self.highexp_seqs_999, 'jaspar')
        # self.motif_count_top_sum = motif_count_top['motif'].value_counts()

    def initialize_reward_split_model(self, mode):
        if mode == 'train':
            model_load = LightningModel.load_from_checkpoint(
                os.path.join(self.base_path, 'mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt'),
                map_location='cpu')
        elif mode == 'eval':
            model_load = LightningModel.load_from_checkpoint(
                os.path.join(self.base_path, 'mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt'),
                map_location='cpu')
        else:
            raise NotImplementedError()
        model_load = model_load.to(self.device)
        model_load.train_params['logger'] = None
        model_load.eval()
        for param in model_load.parameters():
            param.requires_grad = False
        return model_load

    def cal_highexp_kmers(self, k=3, return_clss=False):
        train_set = dna_utils.get_datasets_gosai(self.base_path)
        exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.99)  # 4.56
        highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
        highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
        highexp_seqs = dna_utils.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
        highexp_kmers_99 = count_kmers(highexp_seqs, k=k)
        n_highexp_kmers_99 = len(highexp_indices)

        exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.999)  # 6.27
        highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
        highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
        highexp_seqs = dna_utils.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
        highexp_kmers_999 = count_kmers(highexp_seqs, k=k)
        n_highexp_kmers_999 = len(highexp_indices)

        if return_clss:
            highexp_set_sp_clss_999 = highexp_set_sp.dataset.clss[highexp_set_sp.indices]
            tokens = highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy()
            tokens = torch.tensor(tokens).long().cuda()

            highexp_preds_999 = self.cal_pred_activity(tokens, reward_mode="eval")
            return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs

        return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999

    def cal_pred_activity(self, dna_seqs, reward_mode='eval'):
        """
        :param dna_seqs: token id: bs * len
        :param reward_mode: two reward models
        """

        sample_argmax = 1.0 * F.one_hot(dna_seqs, num_classes=4)
        sample_argmax = torch.transpose(sample_argmax, 1, 2)

        if reward_mode == 'train':
            preds_argmax = self.reward_train(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
        elif reward_mode == 'eval':
            preds_eval = self.reward_eval(sample_argmax).squeeze(-1)
            reward_argmax = preds_eval[:, 0]
        else:
            raise NotImplementedError()
        return reward_argmax  # shape: bs

    def cal_atac_acc(self, dna_seqs):
        sample_argmax = 1.0 * F.one_hot(dna_seqs, num_classes=4)
        sample_argmax = torch.transpose(sample_argmax, 1, 2)

        preds = self.atac_acc(sample_argmax).detach().cpu().numpy().squeeze()  # bs * 7
        return preds[:, 1]
        return (preds[:, 1] > 0.5).sum() / 640

        return preds  # numpy array with shape [n_seqs, 7]

    def compare_kmer(self, kmer2, n_sp2):
        kmer1 = self.highexp_kmers_999
        n_sp1 = self.n_highexp_kmers_999

        # seqs = utils.batch_dna_detokenize(dna_seqs)
        # kmer2 = count_kmers(seqs)
        # n_sp2 = len(seqs)

        kmer_set = set(kmer1.keys()) | set(kmer2.keys())
        counts = np.zeros((len(kmer_set), 2))
        for i, kmer in enumerate(kmer_set):
            if kmer in kmer1:
                counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
            if kmer in kmer2:
                counts[i][0] = kmer2[kmer]

        r_value, p_value = pearsonr(counts[:, 0], counts[:, 1])
        return r_value

    def motif_corr(self, dna_seqs):
        seqs = dna_utils.batch_dna_detokenize(dna_seqs)
        motif_count = scan_sequences(seqs, 'jaspar')
        motif_count_sum = motif_count['motif'].value_counts()

        # motifs_summary = pd.concat(
        #     [self.motif_count_top_sum, motif_count_sum], axis=1)
        # motifs_summary.columns = ['top_data', 'model']
        # motifs_summary.corr(method='spearman')

        all_motifs = motif_count_sum.index.union(self.motif_count_top_sum.index)
        motif_count_sum_aligned = motif_count_sum.reindex(all_motifs, fill_value=0)
        motif_count_top_sum_aligned = self.motif_count_top_sum.reindex(all_motifs, fill_value=0)

        r, p = spearmanr(motif_count_sum_aligned, motif_count_top_sum_aligned)
        return r


def count_kmers(seqs, k=3):
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i: i + k]
            try:
                counts[subseq] += 1
            except KeyError:
                counts[subseq] = 1
    return counts


def batch_agg_count_k_mers(dna_seqs, counts, length, k=3):
    seqs = dna_utils.batch_dna_detokenize(dna_seqs)
    new_counts = count_kmers(seqs, k)
    new_length = len(seqs)

    counts = dict(Counter(counts) + Counter(new_counts))
    length += new_length

    return counts, length

