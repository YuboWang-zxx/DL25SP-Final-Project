from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
        
class Predictor(nn.Module):
    def __init__(self,
                 repr_dim: int,
                 action_dim: int = 2,
                 hidden_dim: int = 512,
                 ball_dim: int = 256):
        super().__init__()
        self.ball_dim = ball_dim
        # MLP: (repr_dim + action_dim) -> hidden_dim -> ball_dim
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ball_dim),
            nn.Tanh()
        )

    def forward(self, repr_prev: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # repr_prev: [B, repr_dim]
        # action:    [B, action_dim]
        x = torch.cat([repr_prev, action], dim=-1)
        delta = self.mlp(x)  # [B, ball_dim]
        # Residual: add to previous ball_repr
        ball_prev = repr_prev[:, :self.ball_dim]
        return delta + ball_prev  # [B, ball_dim]


class Encoder(nn.Module):
    """
    Simple CNN encoder: single-channel image -> repr_dim vector
    """
    def __init__(self,
                 input_channels: int = 1,
                 input_size:     tuple = (65, 65),
                 repr_dim:       int   = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, *input_size)
            out = self.conv(sample)
            flat = out.view(1, -1).size(1)

        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, repr_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        h = self.conv(x)
        return self.fc(h)                    # [B, repr_dim]
                

class JEPAModel(nn.Module):
    def __init__(self,
                 device:                str = "cuda",
                 ball_repr_dim:      int = 256,
                 wall_repr_dim:      int = 256,
                 action_dim:         int = 2,
                 hidden_dim:         int = 512,
                 mask_wall:          bool = False,
                 freeze_ball_encoder: bool = False):
        super().__init__()
        self.device     = device
        self.ball_dim   = ball_repr_dim
        self.wall_dim   = wall_repr_dim
        self.action_dim = action_dim
        self.mask_wall  = mask_wall
        self.hidden_dim = hidden_dim
        # Encoders
        self.ball_encoder = Encoder(1, repr_dim=self.ball_dim).to(device)
        self.wall_encoder = Encoder(1, repr_dim=self.wall_dim).to(device)
        #self.wall_encoder = WallEncoder(1, repr_dim=self.wall_dim).to(device)
        if freeze_ball_encoder:
            for p in self.ball_encoder.parameters():
                p.requires_grad = False

        # Predictor: repr_dim = ball + wall
        repr_dim = self.ball_dim + self.wall_dim
        self.repr_dim = repr_dim  
        self.predictor = Predictor(repr_dim, action_dim, hidden_dim, ball_dim=self.ball_dim).to(device)
        
       
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states:  [B, T, 2, H, W]
        # actions: [B, T-1, action_dim]
        B, T, _, H, W = states.shape
        ball_seq = states[:, :, 0:1]
        wall_seq = states[:, :, 1:2] 

        ball_prev = self.ball_encoder(ball_seq[:,0])
        if self.mask_wall:
            wall_prev = torch.zeros_like(self.wall_encoder(wall_seq[:,0]))
        else:
            wall_prev = self.wall_encoder(wall_seq[:,0])
        repr_prev = torch.cat([ball_prev, wall_prev], dim=-1)  # [B, repr_dim]
        preds = [repr_prev.unsqueeze(1)]                        # list of [B,1,repr_dim]

        # Autoregressive prediction
        for t in range(T-1):
            a_t = actions[:,t]                        # [B,action_dim]
            # Predict next ball
            ball_next = self.predictor(repr_prev, a_t)  # [B,ball_dim]
            # Encode next wall
            if self.mask_wall:
                wall_next = torch.zeros_like(self.wall_encoder(wall_seq[:,t+1]))
            else:
                wall_next = self.wall_encoder(wall_seq[:,t+1])
            # Concat for next step
            repr_prev = torch.cat([ball_next, wall_next], dim=-1)  # [B,repr_dim]
            preds.append(repr_prev.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [B, T, repr_dim]

    def infer(self, init_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # init_states: [B,1,2,H,W]
        # actions:     [B,T-1,action_dim]
        B, _, _, H, W = init_states.shape
        Tm1 = actions.shape[1]

        ball_prev = self.ball_encoder(init_states[:,0,0:1])
        if self.mask_wall:
            wall_prev = torch.zeros_like(self.wall_encoder(init_states[:,0,1:2]))
        else:
            wall_prev = self.wall_encoder(init_states[:,0,1:2])
        repr_prev = torch.cat([ball_prev, wall_prev], dim=-1)
        out = [repr_prev.unsqueeze(0)]                      # [1,B,repr_dim]

        for t in range(Tm1):
            a_t = actions[:,t]
            ball_next = self.predictor(repr_prev, a_t)
            wall_next = wall_prev
            repr_prev = torch.cat([ball_next, wall_next], dim=-1)
            out.append(repr_prev.unsqueeze(0))
        return torch.cat(out, dim=0)  # [T, B, repr_dim]


