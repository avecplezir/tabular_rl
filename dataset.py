import torch
from torch.utils.data import DataLoader

from utils.helper_functions import row_col_to_seq


class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, value_function, states, model, mean=None, std=None):
        self.value_function = value_function
        self.states = states
        self.states_seq = row_col_to_seq(self.states, model.num_cols)
        self.states = (self.states - mean) / std
        self.states_dict = {k: v for k, v in zip(self.states_seq, self.states)}

    def __len__(self):
        return len(self.states_seq) * len(self.states_seq)

    def __getitem__(self, idx):
        s_idx = idx % len(self.states_seq)  # start index
        g_idx = idx // len(self.states_seq)  # goal index
        s = self.states_seq[s_idx]
        g = self.states_seq[g_idx]
        if s == g:
            g = self.states_seq[np.random.randint(len(self.states_seq)) % len(self.states_seq)]

        return self.states_dict[s], self.states_dict[g], s, g,