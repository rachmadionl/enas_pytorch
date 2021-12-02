from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackLSTM(nn.Module):
    """
    """

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super(StackLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = self._build_cell()

    def _build_cell(self):
        cell = []
        for _ in range(self.num_layers):
            cell.append(nn.LSTMCell(self.hidden_size, self.hidden_size))
        cell = nn.ModuleList(cell)
        return cell

    def forward(self, x, prev_h, prev_c):
        next_h, next_c = [], []
        # import pdb; pdb.set_trace()
        for i in range(self.num_layers):
            if i != 0:
                x = next_h[-1]
            curr_h, curr_c = self.cell[i](x, (prev_h[i], prev_c[i]))
            next_h.append(curr_h)
            next_c.append(curr_c)

        return next_h, next_c


class NASController(nn.Module):
    """
    """

    def __init__(self,
                 child_num_layers: int = 6,
                 hidden_size: int = 32,
                 lstm_num_layers: int = 2,
                 num_op: int = 4,
                 skip_target: float = 0.4,
                 temperature: Optional[int] = 5,
                 tanh_constant: Optional[float] = 2.5):
        super(NASController, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.child_num_layers = child_num_layers
        self.hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_op = num_op
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.skip_target = torch.tensor([1 - skip_target, skip_target], dtype=torch.float, device=self.device)
        self.model = self._build_model()
        self.g_emb = nn.Parameter(0.2 * torch.rand(1, self.hidden_size) - 0.1)

        self.model = self.model.to(self.device)

    def _build_model(self):
        model = {}
        model['lstm'] = StackLSTM(self.hidden_size, self.lstm_num_layers)
        model['op_fc'] = nn.Linear(self.hidden_size, self.num_op)
        model['op_emb'] = nn.Embedding(self.num_op, self.hidden_size)
        model['skip_attn1'] = nn.Linear(self.hidden_size, self.hidden_size)
        model['skip_attn2'] = nn.Linear(self.hidden_size, self.hidden_size)
        model['skip_attn3'] = nn.Linear(self.hidden_size, 1)

        model = nn.ModuleDict(model)
        return model

    def _sample_op(self, x, prev_h, prev_c, arc_sample, log_probs, entropys):
        # 1st LSTM of node N
        next_h, next_c = self.model['lstm'](x, prev_h, prev_c)
        prev_h, prev_c = next_h, next_c

        # feed LSTM output to nn.Linear (classify between num_op)
        logit = self.model['op_fc'](next_h[-1])

        # scaled by temperature
        if self.temperature is not None:
            logit /= self.temperature

        # scaled by tanh_constant
        if self.tanh_constant is not None:
            logit /= self.tanh_constant * torch.tanh(logit)

        # get probability
        prob = F.softmax(logit, dim=1)

        # sample op from probability
        op = torch.multinomial(prob, num_samples=1)[0]

        op_config = [int(op.detach().cpu().numpy())]

        # Pass to arc_sample
        arc_sample.append(op_config)

        # Calculate inputs for sample_op
        x = self.model['op_emb'](op.long())

        # Calculate log prob
        log_prob = F.cross_entropy(logit, op)
        log_probs.append(log_prob)

        # Calculate entropys
        entropy = log_prob * torch.exp(log_prob)
        entropys.append(entropy)

        return x, prev_h, prev_c, arc_sample, log_probs, entropys

    def _sample_skip(self, args):
        layer_nth, x, prev_h, prev_c, arc_sample, log_probs, entropys, skip_penaltys, anchors, anchors_w_1 = args

        # 2nd LSTM of node N
        next_h, next_c = self.model['lstm'](x, prev_h, prev_c)
        prev_h, prev_c = next_h, next_c

        if layer_nth == 0:
            x = self.g_emb
            x = x.to(self.device)
            arc_sample.append([])  # no skip on the first layer
        else:
            # Generate logit through series of attention mechanism
            query = torch.tanh(self.model['skip_attn2'](next_h[-1]) + torch.cat(anchors_w_1, dim=0))
            query = self.model['skip_attn3'](query)
            logit = torch.cat([-query, query], dim=1)

            # Scale logit with temperature
            if self.temperature is not None:
                logit /= logit

            # Scale logit with tanh_constant
            if self.tanh_constant is not None:
                logit /= self.tanh_constant * torch.tanh(logit)

            # Generate probability of skip connection
            prob = F.sigmoid(logit)

            # Sample skip connection from probability
            skip = torch.multinomial(prob, num_samples=1)

            # Pass the result to sample_arc
            skip_config = skip.squeeze(dim=1).detach().cpu().numpy().tolist()
            arc_sample.append(skip_config)

            # Calculate log probs
            log_prob = F.cross_entropy(logit, skip.squeeze(dim=1))
            log_probs.append(log_prob)

            # Calculate entropys
            entropy = log_prob * torch.exp(log_prob)
            entropys.append(entropy)

            # Calculate skip penaltys
            skip_penalty = prob * torch.log(prob / self.skip_target)
            skip_penalty = torch.sum(skip_penalty)
            skip_penaltys.append(skip_penalty)

            # Calculate inputs for next time step
            # skip shape:         1, layer_nth (layer_nth > 0)
            skip = torch.reshape(skip, (1, layer_nth))
            # cat_anchors shape:  layer_nth, hidden_size
            cat_anchors = torch.cat(anchors, dim=0)
            x = torch.matmul(skip.float(), cat_anchors)
            # x shape:       1, hidden_size
            x /= 1.0 + torch.sum(skip)

        # append anchors
        anchors.append(next_h[-1])

        # append attention of anchors
        attn_1 = self.model['skip_attn1'](next_h[-1])
        anchors_w_1.append(attn_1)

        return x, prev_h, prev_c, arc_sample, log_probs, entropys, skip_penaltys, anchors, anchors_w_1

    def forward(self):
        # initialize variables that will be returned
        arc_sample = []
        log_probs = []
        entropys = []
        skip_penaltys = []

        # initialize skip properties
        anchors = []
        anchors_w_1 = []

        # initialize prev_h and prev_c
        prev_h = [torch.zeros((1, self.hidden_size), device=self.device)
                  for _ in range(self.lstm_num_layers)]
        prev_c = [torch.zeros((1, self.hidden_size), device=self.device)
                  for _ in range(self.lstm_num_layers)]

        # initialize x
        x = self.g_emb
        x = x.to(self.device)

        for layer_nth in range(self.child_num_layers):
            output_op = self._sample_op(x, prev_h, prev_c, arc_sample, log_probs, entropys)
            x, prev_h, prev_c, arc_sample, log_probs, entropys = output_op

            input_skip = (layer_nth, x, prev_h, prev_c, arc_sample, log_probs,
                          entropys, skip_penaltys, anchors, anchors_w_1)
            output_skip = self._sample_skip(input_skip)
            x, prev_h, prev_c, arc_sample, log_probs, entropys, skip_penaltys, anchors, anchors_w_1 = output_skip

        # Accummulate log_probs
        log_probs = torch.sum(torch.stack(log_probs))

        # Accummulate entropys
        entropys = torch.sum(torch.stack(entropys))

        # Accummulate skip_penaltys
        skip_penaltys = torch.sum(torch.stack(skip_penaltys))

        return arc_sample, log_probs, entropys, skip_penaltys
