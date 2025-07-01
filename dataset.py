from typing import NamedTuple, Optional
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        small=False,
        device="cuda",
        toDevice=True,
    ):
        self.device = device
        self.toDevice=toDevice
        if not small:
            self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
            self.actions = np.load(f"{data_path}/actions.npy")
        else:
            self.states = np.load(f"{data_path}/states0p01.npy", mmap_mode="r")
            self.actions = np.load(f"{data_path}/actions0p01.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        
        states = torch.from_numpy(self.states[i]).float()
        actions = torch.from_numpy(self.actions[i]).float()

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float()
        else:
            locations = torch.empty(0)
        if self.toDevice:
            states=states.to(self.device)
            actions=actions.to(self.device)
            locations=locations.to(self.device)
            
        return WallSample(states=states, locations=locations, actions=actions)

# def create_wall_dataloader(
#     data_path,
#     probing=False,
#     device="cuda",
#     batch_size=64,
#     train=True,
#     small=False,
#     toDevice=True,
    
# ):
#     ds = WallDataset(
#         data_path=data_path,
#         probing=probing,
#         small=small,
#         device=device,
#         toDevice=toDevice
#     )

#     loader = torch.utils.data.DataLoader(
#         ds,
#         batch_size,
#         shuffle=train,
#         drop_last=True,
#         pin_memory=False,
#         num_workers=4
#     )

#     return loader
def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
    small=False,
    toDevice=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        small=small,
        device=device,
        toDevice=toDevice
    )

    loader_kwargs = {}
    if not toDevice:
        loader_kwargs["num_workers"] = 4

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
        **loader_kwargs
    )

    return loader
