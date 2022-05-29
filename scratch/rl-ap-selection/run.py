import random
from datetime import datetime
import numpy as np
from ctypes import *
import argparse
import traceback
from py_interface import *
import torch
from DQN import DQN4Graph, DeepQNetwork, DDQN_PER_4Graph
from replay_memory import make_experience
from torch.utils.tensorboard import SummaryWriter
import os
# The environment (in this example, contain 'a' and 'b')
# shared between ns-3 and python with the same shared memory
# using the ns3-ai model.
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('rssi_ap_0', c_double),
        ('rssi_ap_1', c_double),
        ('rssi_ap_2', c_double),
        ('rssi_ap_3', c_double),
        ('rssi_ap_4', c_double),
        ('rssi_ap_5', c_double),
        ('rssi_ap_6', c_double),
        ('rssi_ap_7', c_double),
        ('rssi_ap_8', c_double),
        ('reward', c_double)
    ]

# The result (in this example, contain 'c') calculated by python
# and put back to ns-3 with the shared memory.
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action', c_uint32)
    ]


def update_state_history(state_history, one_step_state):
    state_history = np.roll(state_history, -1, axis=0)
    state_history[-1] = one_step_state
    return state_history
    
def env_step(data):
    state = np.array([data.env.rssi_ap_0, data.env.rssi_ap_1, data.env.rssi_ap_2, data.env.rssi_ap_3, data.env.rssi_ap_4,
                    data.env.rssi_ap_5, data.env.rssi_ap_6, data.env.rssi_ap_7, data.env.rssi_ap_8])
    reward = data.env.reward
    return state, reward

def get_parser():
    parser = argparse.ArgumentParser(description='training configuration')
    parser.add_argument('--total_time', type=int, default=20000)
    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--enable_tb', type=int, default=1)
    parser.add_argument('--time_interval', type=float, default=0.5)

    return parser

parser = get_parser()
args = parser.parse_args()
print(args)

TOTAL_HISTORY_STEP = 64 # Time series steps
time_counter = 1
# ==================================
# parameter that can adjust
enable_tb = args.enable_tb
is_training = args.is_training
total_time = args.total_time
time_interval = args.time_interval
replace_target_iter = 20
batch_size = 32
lr = 0.01
epsilon = 0.6
# ==================================
nWifis = 9
memory_size = (int)(total_time * 0.2)
state_history = np.zeros([TOTAL_HISTORY_STEP, nWifis])
time_history = []
reward_history = []
action_history = []
trajectory = []
seed = 201492894
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cwd = os.getcwd()

mempool_key = 1234                                          # memory pool key, arbitrary integer large than 1000
mem_size = 4096                                             # memory pool size in bytes
memblock_key = 2333                                         # memory block key, need to keep the same in the ns-3 script

exp = Experiment(mempool_key, mem_size, 'rl-ap-selection', '../../')      # Set up the ns-3 environment
exp.reset()                                                 # Reset the environment
ns3Settings = {
                'nWifis': nWifis,
                'total_time': total_time,
                'time_interval': time_interval,
                'cwd': cwd
            }
model_name = 'DCRQN'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = datetime.now().strftime("%m%d-%H%M")
if enable_tb :
    log_file_name = f'tensor_board_logs/{start_time}_{model_name}_'
    if is_training:
        log_file_name += 'training'
    else :
        log_file_name += 'testing'
    summary_writer = SummaryWriter(log_file_name)

# Create NN
model = DeepQNetwork(
                n_actions=nWifis,
                input_size=nWifis,
                device=device,
                timestep=TOTAL_HISTORY_STEP,
                batch_size=batch_size,
                learning_rate=lr,
                reward_decay=0.6,
                is_training=is_training,
                epsilon=epsilon,
                epsilon_min=0.01,
                max_timestep=total_time/time_interval,
                memory_size=memory_size
        )

try:
    rl = Ns3AIRL(memblock_key, Env, Act)    
                    # Link the shared memory block with ns-3 script
    pro = exp.run(setting=ns3Settings, show_output=True)    # Set and run the ns-3 script (sim.cc)
    action = -1
    while not rl.isFinish():
        with rl as data:
            if data == None:
                break
            # AI algorithms here and put the data back to the action
            # print(f"Python get Env:a={data.env.a}, b={data.env.b}")
            one_step_state, reward = env_step(data) # reward is (prev_state_history, action) feedback
            prev_state_history = state_history
            state_history = update_state_history(state_history, one_step_state)
            obs = np.reshape(state_history, (1, 1, state_history.shape[0], state_history.shape[1]))
            if time_counter > 1:
                # print(f"store: action:{action}, reward:{reward}")
                model.memory.append(prev_state_history, state_history, action, reward)
                action_history.append(action)
                reward_history.append(reward)
                if enable_tb :
                    summary_writer.add_scalar('Loss', model.loss_val, time_counter)
                    summary_writer.add_scalar('Reward', reward, time_counter)
                    summary_writer.add_scalar('Actions', action, time_counter)
            # print("time step: {}".format(time_counter))
            action = model.choose_action(obs)
            data.act.action = action
            time_counter += 1
            if len(model.memory) >= batch_size:
                model.learn()
            if time_counter % replace_target_iter == 0:
                model.replace_target_net_weight()
            print('action: ap{}, eps:{:.2f} loss:{:.2f}\n'.format(action, model.epsilon, model.loss_val))
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
    if is_training:
        torch.save(model.policy_net.state_dict(), f'model_weights/{model_name}')
    del exp
