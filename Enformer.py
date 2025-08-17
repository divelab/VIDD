import copy, os
import diffusion_gosai
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def initialize_gen_model_dna(task, base_path, device, grad=False, pretrained=True):
    if task == "rna" or task == "rna_saluki":
        CKPT_PATH = 'artifacts/RNA_Diffusion:v0/best.ckpt'
        print("CKPT_PATH: ", CKPT_PATH)
        GlobalHydra.instance().clear()
        # Initialize Hydra and compose the configuration
        initialize(config_path="configs_gosai_rna", job_name="load_model")
        cfg = compose(config_name="config_gosai.yaml")
    else:
        # CKPT_PATH = 'artifacts/DNA_Diffusion:v0/last.ckpt'
        CKPT_PATH = os.path.join(base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
        print("CKPT_PATH: ", CKPT_PATH)
        # reinitialize Hydra
        GlobalHydra.instance().clear()
        # Initialize Hydra and compose the configuration
        initialize(config_path="configs_gosai", job_name="load_model")
        cfg = compose(config_name="config_gosai.yaml")

    # Initialize the model
    if pretrained:
        ref_model = diffusion_gosai.Diffusion.load_from_checkpoint(CKPT_PATH, config=cfg, map_location='cpu')
        ref_model = ref_model.to(device)
    else:
        ref_model = diffusion_gosai.Diffusion(config=cfg).to(device)

    if not grad:
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    return ref_model
