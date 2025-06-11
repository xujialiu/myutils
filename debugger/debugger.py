import sys
import runpy
import os
from pathlib import Path
import shlex
import subprocess
import re


def get_cmd_list(cmd):
    def replace_backticks(match):
        command = match.group(1)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    
    cmd = re.sub(r'`([^`]+)`', replace_backticks, cmd)
    result = shlex.split(cmd)
    for idx, i in enumerate(result):
        if i == "\n":
            result.pop(idx)
    return result


def main(cmd):
    current_dir = Path(__file__).parent.resolve()
    os.chdir(current_dir)

    cmd = get_cmd_list(cmd)

    if cmd[0] == "python" or cmd[0] == "python3":
        cmd.pop(0)

    if cmd[0] == "-m":
        cmd.pop(0)
        fun = runpy.run_module
    else:
        fun = runpy.run_path

    sys.argv.extend(cmd[1:])
    fun(cmd[0], run_name="__main__")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # sample command to run the script
    cmd = """
    python train.py \
    --result_root_path ./test_outputs \
    --result_name test_`date +%Y%m%d_%H%M%S` \
    --csv_path /raid0/xujialiu/DDR_seg/ddr_seg_cls.csv \
    --data_path /raid0/xujialiu/DDR_seg/preprocess \
    --finetune /raid0/xujialiu/checkpoints/my_VFM_Fundus_weights.pth \
    --nb_classes_cls 4 \
    --nb_classes_seg 4 \
    --batch_size 32 \
    --input_size 224 \
    --accum_iter 1 \
    --blr 0.001
    """
    
    main(cmd)