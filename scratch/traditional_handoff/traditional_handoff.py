import random
from datetime import datetime
import numpy as np
from ctypes import *
import argparse
import traceback
from py_interface import *
from torch.utils.tensorboard import SummaryWriter
import os
# The environment (in this example, contain 'a' and 'b')
# shared between ns-3 and python with the same shared memory
# using the ns3-ai model.
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('action', c_int),
        ('reward', c_double)
    ]

# The result (in this example, contain 'c') calculated by python
# and put back to ns-3 with the shared memory.
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('empty', c_uint32)
    ]
    
def env_step(data):
    action = data.env.action
    reward = data.env.reward
    return action, reward

def get_parser():
    parser = argparse.ArgumentParser(description='training configuration')
    parser.add_argument('--total_time', type=int, default=20000)
    parser.add_argument('--enable_tb', type=int, default=1)
    parser.add_argument('--time_interval', type=float, default=0.5)

    return parser

parser = get_parser()
args = parser.parse_args()
print(args)

time_counter = 1
# ==================================
# parameter that can adjust
enable_tb = args.enable_tb
total_time = args.total_time
time_interval = args.time_interval
# ==================================
nWifis = 9
reward_history = []
action_history = []
seed = 201492894
np.random.seed(seed)
random.seed(seed)
cwd = os.getcwd()

mempool_key = 1234                                          # memory pool key, arbitrary integer large than 1000
mem_size = 4096                                             # memory pool size in bytes
memblock_key = 2333                                         # memory block key, need to keep the same in the ns-3 script

exp = Experiment(mempool_key, mem_size, 'traditional_handoff', '../../')      # Set up the ns-3 environment
exp.reset()                                                 # Reset the environment
ns3Settings = {
                'nWifis': nWifis,
                'total_time': total_time,
                'time_interval': time_interval,
                'cwd': cwd
            }
model_name = 'traditional_handoff'
start_time = datetime.now().strftime("%m%d-%H%M")
if enable_tb :
    log_file_name = f'../rl-ap-selection/tensor_board_logs/{start_time}_{model_name}_testing'
    summary_writer = SummaryWriter(log_file_name)

try:
    rl = Ns3AIRL(memblock_key, Env, Act)    
                    # Link the shared memory block with ns-3 script
    pro = exp.run(setting=ns3Settings, show_output=True)    # Set and run the ns-3 script (sim.cc)
    while not rl.isFinish():
        with rl as data:
            if data == None:
                break
            # AI algorithms here and put the data back to the action
            action, reward = env_step(data)
            action_history.append(action)
            reward_history.append(reward)
            if enable_tb :
                summary_writer.add_scalar('Reward', reward, time_counter)
                summary_writer.add_scalar('Actions', action, time_counter)
            # print("time step: {}".format(time_counter))
            time_counter += 1
            pass
    pro.wait()                                              # Wait the ns-3 to stop
except Exception as e:
    print("Catch expection")
    print(traceback.format_exc())
finally:
    print('===========================================')
    if len(reward_history) != 0:
        print(f'average reward: {sum(reward_history)/len(reward_history)}')
    else :
        print(f'average reward: 0')
    print('===========================================')
    del exp
