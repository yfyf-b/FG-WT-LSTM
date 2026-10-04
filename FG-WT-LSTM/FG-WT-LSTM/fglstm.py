import torch
import torch.nn as nn
from typing import Dict
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head

class DynamicFrequencyGating(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(DynamicFrequencyGating, self).__init__()


        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency_weights = self.gating_network(x)
        gated_features = x * frequency_weights
        return gated_features


class FG_WT_LSTM(BaseModel):


    def __init__(self, cfg):
        super(FG_WT_LSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        input_dim = self.embedding_net.output_size

        self.frequency_gating = DynamicFrequencyGating(input_dim=input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=False
        )


        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_d = self.embedding_net(data)


        gated_x = self.frequency_gating(x_d)


        gated_x = gated_x.transpose(0, 1)


        lstm_out, (h_n, c_n) = self.lstm(gated_x)


        pred = {'h_n': h_n.transpose(0, 1)}
        pred.update(self.head(self.dropout(lstm_out)))

        return pred