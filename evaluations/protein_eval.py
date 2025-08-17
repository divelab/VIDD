import numpy as np
import os

import esm
import pyrosetta
from pyrosetta import rosetta, get_fa_scorefxn
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
import biotite.structure.io as strucio
from tmtools import tm_align

from evaluations.protein_utils import *


pyrosetta.init(options="-mute all")

_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
TOKENIZER = {aa: i for i, aa in enumerate(ALPHABET)}


def esm_to_ptm(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['ptm'].cpu().tolist()[idx]


def esm_to_plddt(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['mean_plddt'].cpu().tolist()[idx] * 1.0/100


def pdb_to_tm(ori_pdb_file, gen_pdb_file):
    """
    maximize tm score
    :param ori_pdb_file / gen_pdb_file: pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    seq_ori = pose_ori_pdb.sequence()
    seq_gen = pose_gen_pdb.sequence()

    ca_coor_ori = []
    for i in range(1, pose_ori_pdb.total_residue() + 1):
        if pose_ori_pdb.residue(i).has("CA"):
            ca_coord = pose_ori_pdb.residue(i).xyz("CA")
            ca_coor_ori.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_ori.append(pose_ori_pdb.sequence()[i - 1])
    ca_coor_ori = np.array(ca_coor_ori)
    # seq_ori = ''.join(seq_ori)

    ca_coor_gen = []
    for i in range(1, pose_gen_pdb.total_residue() + 1):
        if pose_gen_pdb.residue(i).has("CA"):
            ca_coord = pose_gen_pdb.residue(i).xyz("CA")
            ca_coor_gen.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_gen.append(pose_gen_pdb.sequence()[i - 1])
    ca_coor_gen = np.array(ca_coor_gen)
    # seq_gen = ''.join(seq_gen)

    tm_results = tm_align(ca_coor_ori, ca_coor_gen, seq_ori, seq_gen)
    return tm_results.tm_norm_chain1


def pdb_to_crmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    minimize rmsd, if backbone, only consider N,CA,C
    maximize this function
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    if backbone:
        return -rosetta.core.scoring.bb_rmsd(pose_ori_pdb, pose_gen_pdb)
    else:
        return -rosetta.core.scoring.all_atom_rmsd(pose_ori_pdb, pose_gen_pdb)


def pdb_to_drmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    maximize this function
    """
    atom_gen = pdb_file_to_atomarray(gen_pdb_file)
    atom_ori = pdb_file_to_atomarray(ori_pdb_file)

    if backbone:
        atom_gen = get_backbone_atoms(atom_gen)
        atom_ori = get_backbone_atoms(atom_ori)

    dp = pairwise_distances(atom_gen.coord)
    dq = pairwise_distances(atom_ori.coord)

    return -float(np.sqrt(((dp - dq) ** 2).mean()))


def pdb_to_lddt(ori_pdb_file, gen_pdb_file):
    """
    maximize lddt score
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    lddt = rosetta.core.scoring.lddt(pose_ori_pdb, pose_gen_pdb)
    return lddt


def pdb_to_hydrophobic_score(gen_pdb_file, start_residue_index=None, end_residue_index=None):
    """
    maximize this function
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface exposed
    Typically, minimize hydrophobic score
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    hydrophobic_surf = np.logical_and(
        selection_mask * hydrophobic_mask, sasa(atom_array)
    )

    return 1 - sum(hydrophobic_surf) / (sum(selection_mask * hydrophobic_mask) + 1e-8)


def pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file, start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # generate protein
    atom_array_gen = pdb_file_to_atomarray(gen_pdb_file)
    res_ids_gen = np.unique(atom_array_gen.res_id)
    min_id_gen, max_id_gen = min(res_ids_gen), max(res_ids_gen)

    if start is not None:
        start_gen = res_ids_gen[start]
    else:
        start_gen = min_id_gen
    if end is not None:
        end_gen = res_ids_gen[end - 1] + 1
    else:
        end_gen = max_id_gen + 1

    subprotein_gen = atom_array_gen[
        np.logical_and(
            atom_array_gen.res_id >= start_gen,
            atom_array_gen.res_id < end_gen,
        )
    ]
    sse_gen = annotate_sse(subprotein_gen)

    # original protein
    atom_array_ori = pdb_file_to_atomarray(ori_pdb_file)
    res_ids_ori = np.unique(atom_array_ori.res_id)
    min_id_ori, max_id_ori = min(res_ids_ori), max(res_ids_ori)

    if start is not None:
        start_ori = res_ids_ori[start]
    else:
        start_ori = min_id_ori
    if end is not None:
        end_ori = res_ids_ori[end - 1] + 1
    else:
        end_ori = max_id_ori + 1

    subprotein_ori = atom_array_ori[
        np.logical_and(
            atom_array_ori.res_id >= start_ori,
            atom_array_ori.res_id < end_ori,
        )
    ]
    sse_ori = annotate_sse(subprotein_ori)

    if len(sse_gen) != len(sse_ori):
        raise Exception("Error")
    return np.mean(sse_gen == sse_ori)


def pdb_to_match_define_ss(gen_pdb_file, define_sse: str = "a", start=None, end=None):
    """
    maximize this function
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    res_ids = np.unique(atom_array.res_id)
    min_id, max_id = min(res_ids), max(res_ids)

    if start is not None:
        start = res_ids[start]
    else:
        start = min_id
    if end is not None:
        end = res_ids[end - 1] + 1
    else:
        end = max_id + 1

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse = annotate_sse(subprotein)

    return np.mean(sse == define_sse)


def pdb_to_surface_expose_score(gen_pdb_file, start=None, end=None):
    """
    maximize surface exposure
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    res_ids = np.unique(atom_array.res_id)
    min_id, max_id = min(res_ids), max(res_ids)

    if start is not None:
        start = res_ids[start]
    else:
        start = min_id
    if end is not None:
        end = res_ids[end - 1] + 1
    else:
        end = max_id + 1

    residue_mask = np.array([res_id in list(range(start, end)) for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))

    return sum(surface) / (sum(residue_mask) + 1e-8)


def pdb_to_sasa(gen_pdb_file):
    """
    maximize surface area
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    return sum(sasa(atom_array)) / 1e4


def symmetry_score(gen_pdb_file, starts, ends, all_to_all_protomer_symmetry=False):
    """
    maximize this function
    starts: start position index list
    ends: end position index list
    """
    assert len(starts) == len(ends)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    res_ids = np.unique(atom_array.res_id)

    res_starts, res_ends = [], []
    for i in range(len(starts)):
        start_res_id = res_ids[starts[i]]
        end_res_id = res_ids[ends[i] - 1] + 1
        res_starts.append(start_res_id)
        res_ends.append(end_res_id)

    centers_of_mass = []
    for i in range(len(res_starts)):
        start, end = res_starts[i], res_ends[i]
        backbone_coordinates = get_backbone_atoms(
            atom_array[
                np.logical_and(
                    atom_array.res_id >= start,
                    atom_array.res_id < end,
                )
            ]
        ).coord
        centers_of_mass.append(get_center_of_mass(backbone_coordinates))
    centers_of_mass = np.vstack(centers_of_mass)

    return (
        -float(np.std(pairwise_distances(centers_of_mass)))
        if all_to_all_protomer_symmetry
        else -float(np.std(adjacent_distances(centers_of_mass)))
    )


def pdb_to_globularity_score(gen_pdb_file, start=None, end=None):
    """
    maximize globularity score, make it as a ball
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    res_ids = np.unique(atom_array.res_id)
    min_id, max_id = min(res_ids), max(res_ids)

    if start is None:
        start = [min_id]
        end = [max_id + 1]
    elif isinstance(start, int):
        start = [res_ids[start]]
        end = [res_ids[end - 1] + 1]
    elif isinstance(start, list):
        new_start, new_end = [], []
        for i in range(len(start)):
            new_start.append(res_ids[start[i]])
            new_end.append(res_ids[end[i] - 1] + 1)
        start, end = new_start, new_end

    all_glo = []
    for i in range(len(start)):
        cur_start, cur_end = start[i], end[i]
        backbone = get_backbone_atoms(
            atom_array[
                np.logical_and(
                    atom_array.res_id >= cur_start,
                    atom_array.res_id < cur_end,
                )
            ]
        ).coord

        center_of_mass = get_center_of_mass(backbone)
        m = backbone - center_of_mass
        cur_glo = float(np.std(np.linalg.norm(m, axis=-1)))
        all_glo.append(cur_glo)

    glo = sum(all_glo) / len(all_glo)
    return -glo


def pdb_to_rosetta_energy(gen_pdb_file):
    pose = pose_read_pdb(gen_pdb_file)
    scorefxn = get_fa_scorefxn()

    try:
        total_score = scorefxn(pose)
        return total_score / 1e4
    except Exception as e:
        return 0.0


class ProteinEvalMetrics:
    def __init__(
            self,
            args,
            device,
            result_save_folder="",
    ):
        self.args = args
        self.gen_protein_folder = os.path.join(result_save_folder, 'saved_proteins')
        os.makedirs(self.gen_protein_folder, exist_ok=True)
        self.metrics_list = args.reward.split(",")
        self.metrics_weight = args.reward_weight.split(",")
        self.metrics_weight = [float(x) for x in self.metrics_weight]  # weight for different metrics
        assert len(self.metrics_list) == len(self.metrics_weight)

        # initialize the esmfold model
        if args.folding_model == '3b':
            self.folding_model = esm.pretrained.esmfold_v1().eval()  # esmfold 3b v1
        elif args.folding_model == '650m':
            self.folding_model = esm.pretrained.esmfold_structure_module_only_650M().eval()
        else:
            raise NotImplementedError()
        self.folding_model = self.folding_model.to(device)
        for param in self.folding_model.parameters():
            param.requires_grad = False

    def metrics_cal(
            self,
            metrics_name,
            ori_pdb_file=None,
            gen_pdb_file=None,
            folding_results=None,
            protein_idx=0,
            save_pdb=False,
            pdb_raw=None,
    ):
        all_results = []
        for metric in metrics_name:
            if metric == 'hydrophobic':
                r = pdb_to_hydrophobic_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'match_define_ss':
                r = pdb_to_match_define_ss(gen_pdb_file if save_pdb else StringIO(gen_pdb_file),
                                           define_sse=self.args.define_ss)
            elif metric == 'SASA':
                r = pdb_to_sasa(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'rosseta_energy':
                r = pdb_to_rosetta_energy(gen_pdb_file)
            elif metric == 'symmetry':
                r = symmetry_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file),
                                   starts=self.sym_start, ends=self.sym_end)
            # elif metric == 'affinity':
            #     pass

            elif metric == 'ptm':
                r = esm_to_ptm(folding_results, idx=protein_idx)
            elif metric == 'plddt':
                r = esm_to_plddt(folding_results, idx=protein_idx)
            # elif metric == 'tm':
            #     r = pdb_to_tm(ori_pdb_file, pdb_raw)
            # elif metric == 'crmsd':
            #     # r = pdb_to_crmsd(ori_pdb_file, gen_pdb_file)
            #     r = pdb_to_crmsd(ori_pdb_file, pdb_raw)
            # elif metric == 'drmsd':
            #     r = pdb_to_drmsd(ori_pdb_file, gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            # elif metric == 'lddt':
            #     r = pdb_to_lddt(ori_pdb_file, gen_pdb_file)
            # elif metric == 'match_ss':
            #     # r, _ = pdb_to_match_ss_score(ori_pdb_file if save_pdb else StringIO(gen_pdb_file))
            #     r = pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            #     # r, _ = pdb_to_match_ss_score_original(ori_pdb_file if save_pdb else StringIO(gen_pdb_file))
            # elif metric == 'surface_expose':
            #     r = pdb_to_surface_expose_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'globularity':
                r = pdb_to_globularity_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            else:
                raise NotImplementedError()
            all_results.append(r)
        return all_results

    def reward_metrics(
            self,
            S_sp,
            ori_pdb_file=None,
            save_pdb=False,
            save_pdb_name="",
            return_all_reward_term=False,
    ):
        esm_input_data = []
        for _it, ssp in enumerate(S_sp):
            seq_string = "".join([ALPHABET[x] for _ix, x in enumerate(ssp)])
            esm_input_data.append(seq_string)

        # esmfold forward
        output = self.folding_model.infer(esm_input_data)
        pdbs = self.folding_model.output_to_pdb(output)

        # reward calculation
        record_reward, record_reward_agg = [], []
        each_reward_term = []
        for _it, pdb in enumerate(pdbs):
            if save_pdb:
                cur_protein_name = f"{save_pdb_name}_repeat{_it}"
                # save pdb for potential use
                pdb_path = os.path.join(self.gen_protein_folder, f"{cur_protein_name}.pdb")
                with open(pdb_path, "w") as ff:
                    ff.write(pdb)
                # save fasta for potential use
                fasta_path = os.path.join(self.gen_protein_folder, f"{cur_protein_name}.fasta")
                header = f">{cur_protein_name}"
                with open(fasta_path, 'w') as f:
                    f.write(f"{header}\n")
                    f.write(f"{esm_input_data[_it]}\n")
            else:
                pdb_path = pdb

            # calculate metrics
            all_reward = self.metrics_cal(
                metrics_name=self.metrics_list,
                gen_pdb_file=pdb_path,
                ori_pdb_file=ori_pdb_file,
                folding_results=output,
                protein_idx=_it,
                save_pdb=save_pdb,
            )
            aggregate_reward = sum(v * w for v, w in zip(all_reward, self.metrics_weight))
            # record_reward_agg.append(aggregate_reward)
            record_reward.append(aggregate_reward)
            all_reward_fix = []
            for r_idx, each_reward_name in enumerate(self.metrics_list):
                if each_reward_name == 'hydrophobic':
                    r_fix = 1 - all_reward[r_idx]
                elif each_reward_name == 'globularity':
                    r_fix = -all_reward[r_idx]
                elif each_reward_name == 'symmetry':
                    r_fix = -all_reward[r_idx]
                else:
                    r_fix = all_reward[r_idx]
                all_reward_fix.append(r_fix)
            each_reward_term.append(all_reward_fix)

        if return_all_reward_term:
            return record_reward, each_reward_term
        return record_reward

    def calc_diversity(self, S_sp):
        return set_diversity(S_sp.detach().cpu().numpy())


if __name__ == "__main__":
    import torch
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    glo_pdb = "/data/xsu2/DecodingDistillation/output/protein_plddt,globularity_visualize_glo/saved_proteins/glo1.pdb"
    ss_pdb = "/data/xsu2/DecodingDistillation/output/protein_plddt,match_define_ss_visualize_ss/saved_proteins/ss1.pdb"

    pdb_to_match_define_ss(ss_pdb, define_sse="b")
    pdb_to_globularity_score(glo_pdb)



    random_seq = "AEELSVSRQVIVQDIAYLRSLGYNI"

    folding_model = esm.pretrained.esmfold_v1().eval()
    folding_model = folding_model.to(device)

    output = folding_model.infer(random_seq)
    pdbs = folding_model.output_to_pdb(output)

    # metrics
    # ptm = esm_to_ptm(output)
    # plddt = esm_to_plddt(output)
    # tm = pdb_to_tm(pdb_value_str, pdbs[0])
    # crmsd = pdb_to_crmsd(pdb_value_str, pdbs[0])
    # drmsd = pdb_to_drmsd(StringIO(pdb_value_str), StringIO(pdbs[0]))
    # lddt = pdb_to_lddt(pdb_value_str, pdbs[0])
    # hydrophobic = pdb_to_hydrophobic_score(StringIO(pdbs[0]))
    # match_ss = pdb_to_match_define_ss(StringIO(pdbs[0]))
    # surface_expose = pdb_to_surface_expose_score(StringIO(pdbs[0]))
    # globularity = pdb_to_globularity_score(StringIO(pdbs[0]))
    # pdb_to_rosetta_energy(pdbs[0])

