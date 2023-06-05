import json
import re
from pathlib import Path


if __name__ == "__main__":

    list_exp = []

    # # Large GPUs
    # list_exp.extend(["M4-BiClass-mTAN"])
    # list_exp.extend(["M4-Extrap24H-mTAN"])

    # list_exp.extend(["P12-Extrap24H-IVPVAE", "P12-Extrap24H-IVPVAE-ODE", "P12-Extrap24H-ClassicRNN",
    #                  "P12-Extrap24H-GRUD", "P12-Extrap24H-GOB", "P12-Extrap24H-mTAN",
    #                  "P12-Extrap24H-CRU", "P12-Extrap24H-REDVAE", "P12-Extrap24H-REDVAE-ODE"])

    # list_exp.extend(["P12-BiClass-IVPVAE", "P12-BiClass-IVPVAE-ODE", "P12-BiClass-ClassicRNN",
    #                  "P12-BiClass-GRUD", "P12-BiClass-Raindrop", "P12-BiClass-mTAN",
    #                  "P12-BiClass-REDVAE", "P12-BiClass-REDVAE-ODE"])

    # list_exp.extend(["eICU-BiClass-IVPVAE", "eICU-BiClass-IVPVAE-ODE", "eICU-BiClass-ClassicRNN",
    #                  "eICU-BiClass-GRUD", "eICU-BiClass-Raindrop", "eICU-BiClass-mTAN",
    #                  "eICU-BiClass-REDVAE", "eICU-BiClass-REDVAE-ODE"])

    # list_exp.extend(["eICU-Extrap24H-IVPVAE", "eICU-Extrap24H-IVPVAE-ODE", "eICU-Extrap24H-ClassicRNN",
    #                  "eICU-Extrap24H-GRUD", "eICU-Extrap24H-GOB", "eICU-Extrap24H-mTAN",
    #                  "eICU-Extrap24H-CRU", "eICU-Extrap24H-REDVAE", "eICU-Extrap24H-REDVAE-ODE"])

    # list_exp.extend(["M4-BiClass-IVPVAE", "M4-BiClass-IVPVAE-ODE", "M4-BiClass-ClassicRNN",
    #                  "M4-BiClass-GRUD", "M4-BiClass-Raindrop", "M4-BiClass-REDVAE",
    #                  "M4-BiClass-REDVAE-ODE"])

    # list_exp.extend(["M4-Extrap24H-IVPVAE", "M4-Extrap24H-IVPVAE-ODE", "M4-Extrap24H-ClassicRNN",
    #                  "M4-Extrap24H-GRUD", "M4-Extrap24H-GOB",
    #                  "M4-Extrap24H-CRU", "M4-Extrap24H-REDVAE", "M4-Extrap24H-REDVAE-ODE"])

    list_exp.extend(
        ["eICU-BiClass-IVPVAE", "eICU-BiClass-REDVAE"])

    seeds_start = 5
    seeds_stop = 5

    dr_list = []

    dr_list.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    path_proj = Path(__file__).parents[0]
    file_launch = path_proj/".vscode/launch.json"
    file_gen_args = path_proj/"htcondor_args.txt"

    with open(file_launch, "r") as fin, open(file_gen_args, "a") as fout:
        exp_confs = json.load(fin).get("configurations")
        for exp in exp_confs:
            exp_name = exp.get("name")
            if exp_name in list_exp:
                exp_args = exp.get("args")
                str_exp = ""
                for arg in exp_args:
                    if re.match("--", arg):
                        str_exp += (";" + arg)
                    else:
                        str_exp += ("=" + arg)
                for i in range(seeds_start, seeds_stop + 1):
                    if len(dr_list) > 0:
                        for dr in dr_list:
                            str_args = "--random-state={}".format(
                                i) + str_exp + ";--mask-drop-rate={}".format(dr)
                            print(str_args, file=fout)
                    else:
                        str_args = "--random-state={}".format(i) + str_exp
                        print(str_args, file=fout)
