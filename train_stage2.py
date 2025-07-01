## training
import torch
from tqdm import tqdm
import random
from torch import nn
from dataset import create_wall_dataloader
from models import JEPAModel

from main import load_model

def off_diagonal(x):
    n, m = x.shape
   #assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

def barlow_twin(z1, z2):
    #zi: [B, T, D=HW]
    B, T, D = z1.shape
    BT= B*T
    z1_flatten = z1.view(-1, D)
    z2_flatten = z2.view(-1, D)
    eps=1e-5
    z1_norm = (z1_flatten-z1_flatten.mean(0))/(z1_flatten.std(0) + eps)
    z2_norm = (z2_flatten -z2_flatten.mean(0))/(z2_flatten.std(0) + eps)
    C = torch.matmul(z1_norm.T, z2_norm) /BT
    diag  = torch.diagonal(C).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(C).pow_(2).sum()
    return diag + 0.001 * off_diag


# ## ### 
# def bidirectional_augment(states: torch.Tensor,
#                           actions: torch.Tensor):
   
#     # forward pass (original order)
#     f_states  = states
#     f_actions = actions

#     b_states  = states[-2::-1]=
#     b_actions = -actions.flip(0)

#     aug_states  = torch.cat([f_states,  b_states],  dim=0)
#     aug_actions = torch.cat([f_actions, b_actions], dim=0)
#     return aug_states, aug_actions
####

def train_stage2_model(
    device,
    num_epochs:   int   =30,
    batch_size:   int   = 512,
    lr:           float = 1e-4,
    barlow_lambda: float = 0.001,
):
# load data
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
        toDevice=False,
        batch_size=batch_size
    )

# define model
    ball_dim = 256
    wall_dim = 256
    online_model = JEPAModel(
        device=device,
        ball_repr_dim=ball_dim,
        wall_repr_dim=wall_dim,
        action_dim=2,
        hidden_dim = ball_dim+wall_dim,
        mask_wall=False,
        freeze_ball_encoder = False
    ).to(device)

    online_model.load_state_dict(torch.load('stage1.pth'))
    
    #online_model.ball_encoder.eval()
    
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    for epoch in range(1, num_epochs+1):
        
        online_model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            states, actions =batch.states, batch.actions
            # random cut
            start = random.randint(0, states.size(1)-2)
            end = random.randint(start+1, states.size(1)-1)
            states = states[:, start:end+1].to(device)  # [B, T, 2, H, W]
            actions = actions[:, start:end].to(device)   # [B, T-1, 2]
            # states = states.to(device)
            # actions=actions.to(device)

            preds = online_model(states, actions)        # [B, T, ball+wall]
            z2    = preds[:, :, :ball_dim]         # [B, T, ball_dim]


            B, T, _, H, W = states.shape
            ball_inputs = states[:, :, 0:1].reshape(-1,1,H,W)  # [B*T,1,H,W]
            with torch.no_grad():
                z1_all = online_model.ball_encoder(ball_inputs)
            z1 = z1_all.view(B, T, ball_dim)                   # [B, T, ball_dim]

            mse_loss = mse_criterion(z2[:,1:], z1[:,1:])

            #bt_loss = barlow_twin(z1, z2)
            loss =  mse_loss#bt_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} |MSE={mse_loss:.6f}|  Total={avg_loss:.6f}")

    # save
    torch.save(online_model.state_dict(), "stage2.pth")
    return online_model

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

if __name__ == "__main__":
    dev = get_device()
    train_stage2_model(dev)

