import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax

import torchdiffeq as ode
import utils

_EPS = 1e-10


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if inputs.is_cuda:
            device = inputs.device
            all_msgs = all_msgs.to(device)

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            device = inputs.device
            hidden = hidden.to(device)

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()





class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')
        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_edges):
        # rel_rec [[1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 1. 0.], [0.
        # rel_send [[0. 1. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0.
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims] [128,5,5,4]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            device = single_timestep_inputs.device
            all_msgs = all_msgs.to(device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_edges[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, edges, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        # edges [128,20,2]
        inputs = inputs.transpose(1, 2).contiguous()  # [128,49,5,4]

        sizes = [edges.size(0), inputs.size(1), edges.size(1), edges.size(2)]  # 128，49，20，2
        edges = edges.unsqueeze(1).expand(sizes)  # 128，49，20，2

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_edges = edges[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, curr_edges)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps, preds[0].size(2), preds[0].size(3)]  # [128,49,5,4]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            device = inputs.device
            output = output.to(device)

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()





class ODEDecoder2(nn.Module):
    """ODE decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(ODEDecoder2, self).__init__()
        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')
        self.dropout_prob = do_prob

    def single_step_forward(self, t, single_timestep_inputs):
        # rel_rec [[1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 1. 0.], [0.
        # rel_send [[0. 1. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 0. 1. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.], [1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0.
        # single_timestep_inputs  [batch_size,  num_atoms, num_dims] [128,5,4]
        # single_timestep_edges [batch_size,  num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(self.rel_rec, single_timestep_inputs)
        senders = torch.matmul(self.rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)  # 128，20，8

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            device = single_timestep_inputs.device
            all_msgs = all_msgs.to(device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * self.edges[:, :, i:i + 1]
            all_msgs += msg  # h [batch_size,  num_atoms*(num_atoms-1), num_dims]

        # edge——to——node
        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()  # vi=求和j h（i，j）[128,5,4]

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, edges, rel_rec, rel_send, sample_timesteps, pred_steps, scale):
        # NOTE: Assumes that we have the same graph across all samples.
        # edges [128,20,2]
        inputs = inputs.transpose(1, 2).contiguous()  # [128,49,5,4]

        # edges = edges.unsqueeze(1)  # 128，1，20，2

        sample_timesteps = torch.tensor(sample_timesteps).float().to(inputs.device)
        sample_timesteps = sample_timesteps / scale
        # sample_timesteps.float().to(inputs.device)
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.edges = edges
        preds = []
        for index in range(0, len(sample_timesteps), pred_steps):
            vt = sample_timesteps[index:index + pred_steps + 1]
            if len(vt) != 1:
                pred = ode.odeint(self.single_step_forward, inputs[:, index, :, :], vt, rtol=1e-3,atol=1e-3)  # [pred_step+1,128,5,4]
                pred = pred[1:, :, :, :].transpose(0, 1)
                preds.append(pred)

        output = torch.cat(preds, dim=1)
        return output.transpose(1, 2).contiguous()


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, input_size, hidden_size, num_atoms):
        super(RNNModel, self).__init__()
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.num_atams = num_atoms
        self.input_size = input_size
        self.rnns = nn.ModuleList([nn.LSTM(input_size, input_size, batch_first=True) for _ in range(num_atoms)])

    def forward(self, inputs,prediction_steps):
        # Input shape: [num_sims, num_atoms, num_timesteps, n_in]
        batch_size = inputs.shape[0]
        timesteps = inputs.shape[2]
        outputs = []

        for i in range(self.num_atams):
            ins = inputs[:, i, :, :] # ins shape: [numk_sims,  num_timesteps, n_in]
            outs = []
            state = None
            for step in range(0, timesteps - 1):
                # state = torch.zeros((1, batch_size, self.input_size)).to(inputs.device)
                if step % prediction_steps == 0:
                    x = ins[:, step:step+1, :]
                else:
                    x = outs[step - 1]
                out, state = self.rnns[i](x, state) # gru x :batchsize, timestep(1), hiddensize
                outs.append(out)
            # output, state_new = self.rnns[i](x, state)
            outputs.append(torch.cat(outs, dim=1)) # 5 ,batchsize timestep hiddensize
        outputs=torch.stack(outputs,dim=1)
        return outputs


class RecurrentBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""

    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0.):
        super(RecurrentBaseline, self).__init__()
        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_1 = nn.Linear(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_2 = nn.Linear(n_atoms * n_hid, n_atoms * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        x = x.view(ins.size(0), -1)
        # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        x = x.view(ins.size(0), ins.size(1), -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        x = x + ins

        return x, hidden

    # def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1):
    def forward(self, inputs,prediction_steps):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            # if burn_in:
            #     if step <= burn_in_steps:
            #         ins = inputs[:, :, step, :]
            #     else:
            #         ins = outputs[step - 1]
            # else:
            #     # Use ground truth trajectory input vs. last prediction
            #     if not step % prediction_steps:
            #         ins = inputs[:, :, step, :]
            #     else:
            #         ins = outputs[step - 1]
            if step%prediction_steps==0:
                ins = inputs[:, :, step, :]
            else:
                ins = outputs[step - 1]
            output, hidden = self.step(ins, hidden)
            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs
