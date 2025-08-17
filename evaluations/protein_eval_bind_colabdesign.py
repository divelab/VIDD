import sys, os
import pyrosetta
import jax
import jax.numpy as jnp

from colabdesign import mk_afdesign_model
from colabdesign.af.alphafold.common import residue_constants, protein

from evaluations.protein_utils import *


pyrosetta.init(options="-mute all")

_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
TOKENIZER = {aa: i for i, aa in enumerate(ALPHABET)}


class ProteinEvalMetricsColabDesign:
    def __init__(
            self,
            args,
            device,
            result_save_folder="",
    ):
        self.gen_protein_folder = os.path.join(result_save_folder, 'saved_proteins')
        os.makedirs(self.gen_protein_folder, exist_ok=True)
        self.args = args
        self.device = device
        self.metrics_list = args.reward.split(",")
        self.metrics_weight = args.reward_weight.split(",")
        self.metrics_weight = [float(x) for x in self.metrics_weight]  # weight for different metrics
        assert len(self.metrics_list) == len(self.metrics_weight)

        # binding affinity target proteins
        bind_target_pdb = os.path.join(args.bind_target_folder, f"{args.bind_target}.pdb")

        # model
        self.af2_model = mk_afdesign_model(
            protocol="binder",
            use_multimer=True,
            data_dir="/data484_2/xsu2/ColabDesign/params",
        )
        self.af2_model.prep_inputs(
            pdb_filename=bind_target_pdb,
            chain="A",
            binder_len=args.gen_len,
            rm_binder_seq=False,
        )
        self.target_seq = "".join(residue_constants.restypes_with_x[i] for i in self.af2_model._pdb['batch']['aatype'][:self.af2_model._target_len])

    def reward_metrics(
            self,
            S_sp,
            ori_pdb_file=None,
            save_pdb=False,
            save_pdb_name="",
            return_all_reward_term=False,
    ):
        record_reward, each_reward_term = [], []
        for _it, ssp in enumerate(S_sp):
            seq_string = "".join([ALPHABET[x] for _ix, x in enumerate(ssp)])
            # aatype_binder = np.array(
            #     [residue_constants.restype_order_with_x.get(res, 20) for res in seq_string])
            # self.af2_model._pdb['batch']['aatype'][self.af2_model._target_len:] = aatype_binder
            # self.af2_model._inputs['batch']['aatype'][self.af2_model._target_len:] = aatype_binder
            self.af2_model.predict(seq=seq_string, verbose=False)

            plddt_value = self.af2_model.aux["log"]["plddt"]
            iptm_value = self.af2_model.aux["log"]["i_ptm"]

            # radisug of gration
            ca = self.af2_model.aux['atom_positions'][:, residue_constants.atom_order["CA"]]  # shape: (L, 3)
            ca = ca[-self.af2_model._binder_len:]
            rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
            rg_th = 2.38 * ca.shape[0] ** 0.365
            rg_value = jax.nn.elu(rg - rg_th).item()
            rg_value_max = -rg_value

            aggregate_reward = self.metrics_weight[0] * iptm_value + self.metrics_weight[1] * plddt_value + self.metrics_weight[2] * rg_value_max
            record_reward.append(aggregate_reward)

            each_reward_term.append([iptm_value, plddt_value, rg_value])

            if save_pdb:
                p = protein.Protein(
                    atom_positions=self.af2_model.aux["atom_positions"],
                    aatype=self.af2_model.aux["aatype"],
                    atom_mask=self.af2_model.aux["atom_mask"],
                    residue_index=self.af2_model.aux["residue_index"],
                    b_factors=self.af2_model.aux["plddt"][:, None] * 100 * self.af2_model.aux["atom_mask"],
                )
                pdb_str = protein.to_pdb(p)
                with open("/data/xsu2/DecodingDistillation/test.pdb", "w") as f:
                    f.write(pdb_str)

                cur_protein_name = f"{save_pdb_name}_repeat{_it}"
                # save pdb for potential use
                pdb_path = os.path.join(self.gen_protein_folder, f"{cur_protein_name}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)

        if return_all_reward_term:
            return record_reward, each_reward_term
        return record_reward

    def calc_diversity(self, S_sp):
        return set_diversity(S_sp.detach().cpu().numpy())
