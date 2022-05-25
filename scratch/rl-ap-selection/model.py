import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv

class DCRQN(nn.Module):
    def __init__(self, n_actions):
        super(DCRQN, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2
            )
        self.pool1 = nn.MaxPool2d(
                kernel_size=2,
                padding=1
                )
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            )
        self.pool2 = nn.MaxPool2d(
                kernel_size=2,
                padding=1)
        
        self.rnn1 = nn.RNN(input_size=1632, hidden_size=256)
        self.rnn2 = nn.RNN(input_size=256, hidden_size=256)
        self.fc1 = nn.Linear(in_features=256,out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_actions)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = self.pool1(h1)
        h3 = F.relu(self.conv2(h2))
        h4 = self.pool2(h3)       
        h5 = h4.view(1, h4.shape[0], 32*17*3) # RNN INPUT: tensor of shape (seq_length, batch size, input_size)
        h6, hid_s1 = self.rnn1(h5, ) # RNN OUTPUT: (seq_length, batch_size, D=1)
        h7, hid_s2 = self.rnn2(h6, hid_s1)
        h8 = F.relu(self.fc1(h7))
        out = F.relu(self.fc2(h8))
        # out = self.softmax(h9)
        # print('**************************************************')
        # print(f'q_values: {out}')
        return out

class GCN(torch.nn.Module):
    def __init__(self, in_features, num_of_nodes, batch_size, is_training):
        super().__init__()
        self.batch_size = batch_size
        self.num_of_nodes = num_of_nodes
        self.last_graph_out_features = 32
        self.is_training = is_training
        # self.lstm_i_dim = self.last_graph_out_features*num_of_nodes # input dimension of LSTM
        # self.lstm_h_dim = 256     # output dimension of LSTM
        # self.lstm_N_layer = 1   # number of layers of LSTM
        self.conv1 = GCNConv(in_features, self.last_graph_out_features)
        # self.conv1_bn = BatchNorm(32)
        self.conv2 = GCNConv(64, 64)
        # self.conv2_bn = BatchNorm(16)
        # self.conv3 = GCNConv(64, self.last_graph_out_features) # output: |V|*out_channels
        # self.conv3_bn = BatchNorm(16)
        # self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        # self.lstm2 = nn.LSTM(input_size=self.lstm_h_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(in_features=self.last_graph_out_features*self.num_of_nodes, out_features=128) # output: (*, out_channels)
        self.fc2 = nn.Linear(in_features=128, out_features=num_of_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(f'x.shape, {x.shape}')
        # print(f'edge_index, {edge_index.shape}')
        h1 = F.dropout(F.relu(self.conv1(x, edge_index)), training=bool(self.is_training))

        print(h1)
        # h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), training=bool(self.is_training))
        # h3 = F.dropout(F.relu(self.conv3(h2, edge_index)), training=bool(self.is_training))
        # print(f'h3.shape:{h3.shape}')
        h2_flat = h1.view(-1, self.last_graph_out_features*self.num_of_nodes)
        # print(h3_flat.shape)
        # h4, h4_hidden = self.lstm(h3_flat, )
        # h5, h5_hidden = self.lstm2(h4, h4_hidden)
        h3 = F.relu(self.fc1(h2_flat))
        out = F.relu(self.fc2(h3))
        print(f'q_values:{out}')
        return out

class GAT(torch.nn.Module):
    def __init__(self, in_features, num_of_nodes, batch_size, is_training):
        super().__init__()
        self.batch_size = batch_size
        self.num_of_nodes = num_of_nodes
        # self.last_graph_out_features = 32
        self.last_graph_out_features = 128

        self.is_training = is_training
        # self.lstm_i_dim = self.last_graph_out_features*num_of_nodes # input dimension of LSTM
        # self.lstm_h_dim = 256     # output dimension of LSTM
        # self.lstm_N_layer = 1   # number of layers of LSTM
        self.conv1 = GATConv(in_features, 128, dropout=0.5)
        self.conv2 = GATConv(128, 128, dropout=0.5)
        self.conv3 = GATConv(128, self.last_graph_out_features, dropout=0.5)

        # self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        # self.lstm2 = nn.LSTM(input_size=self.lstm_h_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(in_features=self.last_graph_out_features*self.num_of_nodes, out_features=128) # output: (*, out_channels)
        self.fc2 = nn.Linear(in_features=128, out_features=num_of_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(f'x.shape, {x.shape}')
        # print(f'edge_index, {edge_index.shape}')
        h1 = F.relu(self.conv1(x, edge_index))
        # print(h1)
        h2 = F.relu(self.conv2(h1, edge_index))
        h3 = F.relu(self.conv3(h2, edge_index))
        # print(f'h1.shape:{h1.shape}')
        h2_flat = h1.view(-1, self.last_graph_out_features*self.num_of_nodes) # (batch_size, flatten_size)
        # print(h2_flat.shape)
        # h4, h4_hidden = self.lstm(h3_flat, )
        # h5, h5_hidden = self.lstm2(h4, h4_hidden) d
        h3 = F.relu(self.fc1(h2_flat))
        out = F.relu(self.fc2(h3))
        # print(f'q_values:{out}')
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(4, 3, 3).to(device)
    data = torch.rand([3, 4]).to(device), torch.tensor([[0,1,1,2], [1,0,2,1]]).to(device)
    print(data[0].shape)
    print(data[1].shape)
    out = model(data)
    print(out)