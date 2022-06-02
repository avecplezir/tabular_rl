import torch
from torch import nn
import copy
import numpy as np

from utils.helper_functions import row_col_to_seq


class ValueApproximator(nn.Module):
    def __init__(self, middle_dim=4, states=None, mean=0, std=1, device='cuda', value_function=None,
                 model=None):
        super().__init__()
        self.estimate_value = nn.Sequential(
            nn.Linear(4, middle_dim),
            nn.LeakyReLU(),
            nn.Linear(middle_dim, 1),
            #               nn.LeakyReLU(),
            #               nn.Linear(64, 1)
        )

        middle_proba_dim = 16
        self.middle_state = nn.Sequential(
            nn.Linear(2 * 3, middle_proba_dim),
            nn.LeakyReLU(),
            #             nn.Linear(middle_proba_dim, middle_proba_dim),
            #             nn.LeakyReLU(),
            nn.Linear(middle_proba_dim, 2),
        )

        self.states_original = states
        self.mean = mean
        self.std = std
        self.states = torch.tensor((self.states_original - mean) / std).float().to(device)
        self.value_function = torch.tensor(value_function).float().to(device)
        self.cos = torch.nn.CosineSimilarity()
        self.temperature = 0.07
        self.recursive_depth = 0
        self.model = model

    def get_x(self, s, g):
        noise = torch.randn_like(s)
        x = torch.cat([s, g, noise], 1)
        return x

    def sample_middle_state(self, s, g):
        x = self.get_x(s, g)
        m = self.middle_state(x)
        sim = self.cos(m.repeat(self.states.size(0), 1), self.states)
        p = torch.nn.functional.gumbel_softmax(sim, tau=self.temperature, hard=True)
        m = (p.unsqueeze(-1) * self.states).sum(0)
        m_idx = row_col_to_seq(m[None], self.model.num_cols).long()
        return m.unsqueeze(0), m_idx

    def update_value_table(self, state, goal):
        P = copy.deepcopy(self.model.P)
        P[goal, :, :] = 0
        P[goal, self.model.num_states - 1, :] = 1
        # store old value
        tmp = self.value_function[state, goal].copy()
        # compute the value function
        self.value_function[state, goal] = np.max(
            np.sum((self.model.R[state, goal] + self.model.gamma * self.value_function[:, goal, :]) * P[state, :, :], 0))
        delta = np.abs(tmp - self.value_function[state, goal])
        return delta

    def loss(self, output, target):
        return torch.abs(output - target).mean()

    def forward(self, s, g, s_idx, g_idx):
        data = self.recursive_forward(s, g, s_idx, g_idx)
        self.recursive_depth = 0
        return data

    def recursive_forward(self, s, g, s_idx, g_idx):
        self.recursive_depth += 1
        m, m_idx = self.sample_middle_state(s, g)
        if m_idx == g_idx or m_idx == s_idx or self.recursive_depth > 2:
            v_sg = self.value_function[s_idx.detach(), g_idx.detach()]
            y_sg = self.estimate_value(torch.cat([s, g], 1))
            loss = self.loss(y_sg, v_sg)
            return y_sg, v_sg, loss
        else:
            y_sm, v_sm, loss_sm = self.forward(s, m, s_idx.detach(), m_idx.detach())
            y_mg, v_mg, loss_mg = self.forward(m, g, m_idx.detach(), g_idx.detach())
            v_sg = self.value_function[s_idx.detach(), g_idx.detach()]
            #             loss_sg = self.loss(v_sm+v_mg, v_sg) + loss_sm + loss_mg
            loss_sg = loss_sm + loss_mg
            return y_sm + y_mg, v_sm + v_mg, loss_sg
