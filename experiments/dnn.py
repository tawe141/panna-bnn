import torch
from torch.utils.data import DataLoader
from data.dataset import PANNADiamineDataset, InMemoryPANNA
from utils.data import custom_collate
import tqdm
from math import sqrt
import wandb
wandb.init(project='diamine-nn-potential', entity='tawe141')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
CONFIGURATION
"""
BATCH_SIZE = 512
LR = 0.2
D_H = 16

config = wandb.config
config.learning_rate = LR
config.batch_size = BATCH_SIZE
config.hidden_size = D_H


class DNN(torch.nn.Module):
    def __init__(self, n_atoms=5):
        super(DNN, self).__init__()
        self.ff = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(360, D_H),
                torch.nn.BatchNorm1d(D_H),
                torch.nn.ReLU(),
                torch.nn.Linear(D_H, D_H),
                # torch.nn.BatchNorm1d(D_H),
                torch.nn.ReLU(),
                # torch.nn.Linear(D_H, D_H),
                # torch.nn.BatchNorm1d(D_H),
                # torch.nn.ReLU(),
                torch.nn.Linear(D_H, 1)
            )
            for _ in range(n_atoms)
        ])

    def forward(self, x: torch.FloatTensor, species_idx: torch.LongTensor, batch_idx: torch.LongTensor):
        atomic_energy_contributions = torch.zeros_like(species_idx, device=device, dtype=torch.float32)
        for i in range(len(self.ff)):
            atomic_energy_contributions[species_idx == i] = self.ff[i](x[species_idx == i]).squeeze()

        cohesive_energies = torch.zeros(batch_idx[-1]+1, device=device)
        cohesive_energies.scatter_add_(0, batch_idx, atomic_energy_contributions)
        return cohesive_energies


model = DNN()
if torch.cuda.is_available():
    model = model.cuda()

dataset = InMemoryPANNA(standardize_x=True, standardize_y=False)
train_size = int(len(dataset) * 0.9)
train_set, test_set = torch.utils.data.random_split(dataset, lengths=(train_size, len(dataset)-train_size))
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=custom_collate, pin_memory=True, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5)
loss_fn = torch.nn.MSELoss()

epochs = 500

if __name__ == "__main__":
    wandb.watch(model, log_freq=1, log='all')
    model.train()

    for e in range(epochs):
        with tqdm.tqdm(train_dataloader) as pbar:
            for batch in pbar:
                x, species_idx, y, batch_idx = batch
                if torch.cuda.is_available():
                    x, species_idx, y, batch_idx = x.cuda(), species_idx.cuda(), y.cuda(), batch_idx.cuda()
                optimizer.zero_grad()
                # Output from model
                output = model(x, species_idx, batch_idx)
                # Calc loss and backprop gradients
                loss = loss_fn(output, y)
                pbar.set_description('LR: %.1e, Epoch %i / %i; MSE: %.3f, RMSE/atom: %.2e' % (scheduler.get_last_lr()[0], e, epochs, loss.item(), loss.item() / len(species_idx) * len(y)))
                loss.backward()
                optimizer.step()
            wandb.log({
                'loss': loss
            })
            scheduler.step()

    # with torch.no_grad():
    #     dataloader = DataLoader(test_set, batch_size=512, num_workers=2, pin_memory=True)
    #     predictions = []
    #     y_true = []
    #     for batch in tqdm.tqdm(dataloader):
    #         x, y = batch
    #         if torch.cuda.is_available():
    #             x = x.cuda()
    #             # y = y.cuda()
    #         predictions.append(mlp(x).cpu().flatten())
    #         y_true.append(y)
    #     mse = torch.nn.functional.mse_loss(torch.cat(predictions), torch.cat(y_true))
    #     print('Test RMSE: %f' % torch.sqrt(mse))