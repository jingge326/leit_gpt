import json
import re
from pathlib import Path


if __name__ == "__main__":

    list_exp = []
    list_exp.extend(
        ["P12-BiClass-GPTS"])

    seeds_start = 1
    seeds_stop = 5

    dr_list = []

    # dr_list.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

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
