import sys
import runpy
import os

os.chdir("/data_A/xujialiu/projects/0_personal/my_utils/debugger")

cmd = 'python test.py'

cmd = cmd.split()
if cmd[0] == "python":
    """pop up the first in the args"""
    cmd.pop(0)

if cmd[0] == "-m":
    """pop up the first in the args"""
    cmd.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(cmd[1:])


fun(cmd[0], run_name="__main__")