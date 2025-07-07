import os
import glob
import shutil
import numpy as np
import torch
from tqdm import tqdm
import sys

def convert_opt_weights(model_name, orignial_path, output_path):

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    bin_files = glob.glob(os.path.join(orignial_path, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(output_path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))

if __name__ == "__main__":
    args = sys.argv[1:]
    convert_opt_weights(args[0], args[1], args[2])