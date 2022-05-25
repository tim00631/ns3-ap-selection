from collections import namedtuple
import os
import random
import re
import time
from mininet.term import makeTerm
import numpy as np
from mn_wifi.mobility import Mobility as mob
import torch
from torch_geometric.data import Data

def get_trace(sta, file_):
    file_ = open(file_, 'r')
    raw_data = file_.readlines()
    file_.close()

    sta.p = []
    pos = (-1000, 0, 0)
    sta.position = pos

    for data in raw_data:
        line = data.split()
        x = line[0]  # First Column
        y = line[1]  # Second Column
        pos = float(x), float(y), 0.0
        sta.p.append(pos)
    return sta.p

def write_stats_by_file_name(data, file_name):
    create_file_if_not_exist(file_name)
    with open(file_name, 'w') as f:
            for d in data:
                f.write(str(d)+'\n')

def store_position(dict_list, file_name, type, start):
    create_file_if_not_exist(file_name)
    with open(file_name, 'w') as f:
            for i in range(len(dict_list)):
                pos = dict_list[i]['position']
                f.write(f'{type}{start+i}:{pos}\n')

def create_file_if_not_exist(file_name):
    if not os.path.exists(file_name):
        os.mknod(file_name)
        os.system(f"chmod +rw {file_name}")

def gen_random_number_between(min, max):
    return random.randint(min, max)

def _parseIperf(iperfOutput, l4Type='UDP'):
    if l4Type == 'UDP':
        r = r'Bytes +(.*) Mbits/sec'
    else :
        r = r'Bytes +(.*) Mbits/sec'
    m = re.findall(r, iperfOutput)
    if m:
        return m[-1]
    else:
        # was: raise Exception(...)
        print(('could not parse iperf output: ' + iperfOutput))
        return 0

def iperf(hosts=None, l4Type='TCP', udpBw='600M', iperf_interval=1, port=32121):
    server_file = 'log/iperf_server.log'
    client_file = 'log/iperf_client.log'
    # os.system(f"rm -rf log/*.log")
    os.system(f'rm -rf {server_file}')
    os.system(f'rm -rf {client_file}')
    create_file_if_not_exist(server_file)
    create_file_if_not_exist(client_file)

    hosts = hosts or [hosts[0], hosts[-1]]
    assert len(hosts) == 2
    client, server = hosts

    server.cmd('killall -9 iperf3')
    iperfArgs = f'iperf3 -p {port} -f m'
    serverProc = makeTerm(server, cmd=f'{iperfArgs} -s --logfile {server_file}')
    # print(f'run server cmd: {iperfArgs} -s')
    # print(f'run client cmd: {iperfArgs} -t {time_interval} -c {server.IP()} {bwArgs}')
    clientProc = makeTerm(client, cmd=f'{iperfArgs} -u -b {udpBw} -t {iperf_interval} -c {server.IP()} --logfile {client_file}')
    cliout, err = clientProc[0].communicate()
    # clientProc[0].terminate()
    serOut = ''
    cliOut = ''
    if os.path.exists(client_file):
        with open(client_file, 'r') as f:
            for line in f.readlines():
                cliOut += line
    if os.path.exists(server_file):
        with open(server_file, 'r') as f:
            for line in f.readlines():
                serOut += line
    # print(serOut)
    print(cliOut)
    result = float(_parseIperf(cliOut, l4Type=l4Type))
    return result

def step(node, action, ap_map, aps, hosts):
    '''
    node: target STA
    action: target AP (handover)

    return reward
    '''
    target_ap = get_ap_name(action, ap_map)
    in_range_list = []

    for k in node.wintfs[0].apsInRange.keys():
        in_range_list.append(str(k))

    if target_ap not in in_range_list:
        print(f'{target_ap} is not in the range of STA')
        return 0
    else:
        do_handover(node, action, aps)
        success = check_connection(node, action, aps, 10)
        if success:
            reward = iperf([node, hosts[action]])
        else :
            print("retry timeout, cannot associate to this ap due to some problems...")
            output = node.cmd('iw dev sta1-wlan0 scan')
            print(output)
            reward = 0
        return reward

def get_state(target_node, bg_stas, aps, n_bg_stas, n_aps, ap_map):
    update_rssi_map([target_node]+bg_stas, aps)
    bg_noise = -92
    interference_threshold = -72
    offset = 93
    bg_signal = np.full([n_bg_stas, n_aps],fill_value=-93)
    rssi_map = np.full(n_aps, fill_value=-93)
    # state = np.full(NUM_OF_APS, fill_value=-91)
    # bg_signal = np.zeros([NUM_OF_BG_STAS, NUM_OF_APS])
    bg_rssi_with_filtered = np.zeros([n_bg_stas, n_aps])
    # rssi_map = np.zeros(NUM_OF_APS)
    state = np.zeros(n_aps)
    for i in range(len(bg_stas)):
        for ap_name, signal in bg_stas[i].wintfs[0].apsInRange.items():
            if signal > interference_threshold:
                bg_rssi_with_filtered[i, ap_name_get_index(ap_name, ap_map)] = signal+offset
            bg_signal[i, ap_name_get_index(ap_name, ap_map)] = signal
    # print(bg_rssi_with_filtered)
    bg_signal_sum = np.sum(bg_rssi_with_filtered, axis=0)
    # print(bg_signal_sum)
    # print(bg_signal_sum + np.array(bg_noise)+np.array(offset))
    for ap_name, signal in target_node.wintfs[0].apsInRange.items():
        rssi_map[ap_name_get_index(ap_name, ap_map)] = signal
    state = (rssi_map + np.array(offset))/(bg_signal_sum+(np.array(bg_noise)+np.array(offset)))
    # state = rssi_map / (bg_signal_sum + np.array(bg_noise))

    return state, bg_signal, rssi_map

def do_handover(node, action, aps):
    # print(f'{node} to {aps[action]}')

    if node.wintfs[0].associatedTo == None:
            node.wintfs[0].associate_infra(aps[action].wintfs[0])
    else:
        if aps[action].wintfs[0] == node.wintfs[0].associatedTo:
            # print("target_ap is the same one")
            pass
        else:
            node.wintfs[0].disconnect(node.wintfs[0].associatedTo)
            node.wintfs[0].associate_infra(aps[action].wintfs[0])

def check_connection(node, action, aps, max_retry_time):
    # print(f"expect node to {str(aps[action])},\n")
    retry = 1
    s = node.cmd(f'iw dev {node.wintfs[0]} link')
    match = re.findall(r'SSID: [\w]+', s)
    if match:
        match = match[0][-3:]
    while match != str(aps[action]):
        if retry == max_retry_time:
            return False # cannot associate to this ap
        s = node.cmd(f'iw dev {node.wintfs[0]} link')
        retry += 1
        match = re.findall(r'SSID: [\w]+', s)
        if match:
            match = match[0][-3:]
        time.sleep(1)
    # print(f"get: {s}")
    return True

def ap_name_get_index(ap_name, ap_map):
    """
    input: ap_name
    return corresponding index of ap_map
    """
    if ap_map.get(str(ap_name)) != None:
        return ap_map.get(str(ap_name))
    else:
        return -1

def get_ap_name(target_i, ap_map):
    """
    input: action index
    return:ap name
    """
    for k, v in ap_map.items():
        if v == target_i:
            return k

def get_ap_index_by_associatedTo(node):
    return int(str(node.wintfs[0].associatedTo)[2])-1

def update_rssi_map(nodes, aps):
    ap_nodes = aps
    aps = []
    for node in nodes:
        for ap in ap_nodes:
            dist = node.get_distance_to(ap)
            if dist > ap.wintfs[0].range:
                mob.ap_out_of_range(mob, node.wintfs[0], ap.wintfs[0])
            else :
                mob.ap_in_range(mob, node.wintfs[0], ap, dist)

    for ap in aps:
        dist = node.get_distance_to(ap)

def update_state_history(state_history, state):
    state_history = np.roll(state_history, -1, axis=0)
    state_history[-1] = state
    return state_history

def generate_traffic(bg_stas, hosts, total_timestep, time_interval, pkt_size=1024, rate=1024*16):
    # shutdown problem rate=1024*32*16
    process = []
    os.system("cd log/ && rm -rf *.log")
    for host in hosts:
        # print(f"ITGRecv for {host}, ip: {host.IP()}")
        process.append(makeTerm(host, cmd='ITGRecv'))
    time.sleep(3)
    for i in range(len(bg_stas)):
        j = get_ap_index_by_associatedTo(bg_stas[i])
        # cmd = f'ITGSend -T TCP -a {hosts[j].IP()} -c {pkt_size} -C {rate} -t {1e7*1000*total_timestep*time_interval} -l log/sender_{bg_stas[i]}.log -x log/receiver_{hosts[j]}.log'
        cmd = f'ITGSend -T UDP -a {hosts[j].IP()} -c {pkt_size} -C {rate} -t {2147483647}'
        print(f'{bg_stas[i]} cmd: {cmd}')
        bg_stas[i].sendCmd(cmd)
    return process

def parse_graph(sta1, state_history, aps: list, ap_map) -> Data:
    # state_history [64,9]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(state_history.transpose(),dtype=torch.float).to(device) # [9, 64]
    # print(x)
    edge_index = [[],[]]

    # in_range fully connected
    # in_range_list = []
    # for k in sta1.wintfs[0].apsInRange.keys():
    #     in_range_list.append(ap_name_get_index(str(k), ap_map))
    # for i in in_range_list:
    #     for j in in_range_list:
    #         if i != j:
    #             edge_index[0].append(i)
    #             edge_index[1].append(j)
    # complete graph
    for i in range(len(aps)):
        for j in range(len(aps)):
            if aps[i] != aps[j]:
                edge_index[0].append(i)
                edge_index[1].append(j)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    # print(edge_index)

    graph = Data(x=x, edge_index=edge_index)
    return graph