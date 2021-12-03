from typing import Optional
import torch

from controller_model import NASController


class Controller:
    def __init__(self,
                 child_num_layers: int = 6,
                 hidden_size: int = 32,
                 lstm_num_layers: int = 2,
                 num_op: int = 4,
                 skip_target: float = 0.4,
                 skip_weight: float = 0.8,
                 temperature: Optional[int] = 5,
                 tanh_constant: Optional[float] = 2.5,
                 entropy_weight: float = 0.0001,
                 baseline_decay: float = 0.999,
                 lr: float = 0.0002,
                 train_step_num: int = 50,
                 sample_size: int = 5
                 ):
        """
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = NASController(child_num_layers, hidden_size, lstm_num_layers,
                                 num_op, skip_target, temperature, tanh_constant)
        self.net = self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr)

        self.skip_weight = skip_weight
        self.entropy_weight = entropy_weight
        self.baseline_decay = baseline_decay
        self.train_step_num = train_step_num
        self.sample_size = sample_size

        self.baseline = torch.zeros(1).to(self.device)

    def fit(self, child_model, valid_loader):

        # Amount of training steps in one epoch
        for step in range(self.train_step_num):
            losses = torch.zeros(self.sample_size, device=self.device)

            # Amount of models (data) sampled in one train step
            for sample in range(self.sample_size):
                # sample a model
                arc_sample, log_probs, entropys, skip_penaltys = self.net()

                # Valid a model from sampled arch
                rewards = child_model.valid_rl(arc_sample, valid_loader)
                rewards += entropys * self.entropy_weight

                # Update baseline
                with torch.no_grad():
                    self.baseline = self.baseline + (1 - self.baseline_decay) * (rewards - self.baseline)

                # Update loss
                losses[sample] = log_probs * (rewards - self.baseline) + self.skip_weight * skip_penaltys

            loss = losses.sum() / self.sample_size

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print(f'step: {step:2d}\tloss: {loss:.5f}')

    def sample_arc(self):
        outputs = self.net()
        arc_sample = outputs[0]

        return arc_sample

    def valid(self, child_model, arc_num: int, valid_loader):

        accuracys = []
        arcs = []
        for _ in range(arc_num):
            outputs = self.net()
            arc_sample = outputs[0]

            accuracy = child_model.valid(arc_sample, valid_loader)

            arcs.append(arc_sample)
            accuracys.append(accuracy)

        for i in range(len(arcs)):
            print(f'arc: {arcs[i]}\tacc: {accuracys[i]:.5f}')

        return accuracys

    def get_best_arc(self, child_model, arc_num: int, valid_loader):
        self.net.eval()

        best_accuracy = 0
        accuracys = []
        for _ in range(arc_num):
            outputs = self.net()
            arc_sample = outputs[0]

            accuracy = child_model.valid(arc_sample, valid_loader)
            accuracys.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_arc = arc_sample

        print(accuracys)
        print(best_accuracy)
        print(best_arc)
        return best_accuracy, best_arc
