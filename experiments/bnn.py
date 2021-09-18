import torch
from torch.utils.data import DataLoader
from data.dataset import PANNADiamineDataset, InMemoryPANNA
from utils.data import custom_collate
import tqdm
import pyro
from pyro.contrib.bnn.hidden_layer import HiddenLayer
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.contrib.autoname import name_count
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
CONFIGURATION
"""
BATCH_SIZE = 1024
LR = 0.002
D_H = 16
RECORD = True
# GAMMA = 0.5
# LR_STEP_MILESTONES = [200, 400]

if RECORD:
    wandb.init(project='diamine-nn-potential', entity='tawe141')

    config = wandb.config
    config.model_type = 'BNN'
    config.learning_rate = LR
    config.batch_size = BATCH_SIZE
    config.hidden_size = D_H
# config.gamma = GAMMA
# config.lr_step_milestones = LR_STEP_MILESTONES


class FeedForwardBNN(PyroModule):
    def __init__(self):
        super(FeedForwardBNN, self).__init__()
        # print('Reported output variables distributed as N(%.2f, %.2f)' % (train_mean, train_variance))

        # define the same prior on weights + biases
        mu = torch.Tensor([0.0])
        std = torch.Tensor([1.0])
        if torch.cuda.is_available():
            mu = mu.cuda()
            std = std.cuda()

        self.fc1 = PyroModule[torch.nn.Linear](360, D_H)
        self.fc1.weight = PyroSample(dist.Normal(mu, std).expand([D_H, 360]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(mu, std).expand([D_H]).to_event(1))

        # batchnorm; note that Pyro also has its own batchnorm (https://docs.pyro.ai/en/dev/distributions.html#batchnorm)
        # but I have no idea how it works... crossing my fingers that this does fine
        self.bn = PyroModule[torch.nn.BatchNorm1d](D_H)
        self.bn.weight = PyroSample(dist.Normal(mu, std).expand([D_H]).to_event(1))
        self.bn.bias = PyroSample(dist.Normal(mu, std).expand([D_H]).to_event(1))

        self.fc2 = PyroModule[torch.nn.Linear](D_H, D_H)
        self.fc2.weight = PyroSample(dist.Normal(mu, std).expand([D_H, D_H]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(mu, std).expand([D_H]).to_event(1))

        # self.fc3 = PyroModule[torch.nn.Linear](D_H, D_H)
        # self.fc3.weight = PyroSample(dist.Normal(mu, std).expand([D_H, D_H]).to_event(2))
        # self.fc3.bias = PyroSample(dist.Normal(mu, std).expand([D_H]).to_event(1))

        self.fc3 = PyroModule[torch.nn.Linear](D_H, 1)
        self.fc3.weight = PyroSample(dist.Normal(mu, std).expand([1, D_H]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(mu, std).expand([1]).to_event(1))

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # x = x.reshape(-1, 1)
        x = self.activation(self.bn(self.fc1(x)))
        x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        return self.fc3(x).squeeze()

        # sigma = pyro.sample('sigma', dist.Uniform(torch.Tensor([0.0]).cuda(), torch.Tensor([1.0]).cuda()))
        # with pyro.plate('data', x.shape[0], use_cuda=torch.cuda.is_available()):
        #     obs = pyro.sample('obs', dist.Normal(mu, sigma), obs=y)
        # return mu  # return obs here?

#
# class FeedForwardHidden(PyroModule):
#     # basically the same as `FeedForwardBNN`, but with `HiddenLayer` in Pyro
#     def __init__(self):
#         super().__init__()
#         self.fc1 = HiddenLayer()



class BNN(PyroModule):
    def __init__(self, n_species=5):
        super(BNN, self).__init__()
        self.ff = torch.nn.ModuleList([
            FeedForwardBNN() for _ in range(n_species)
        ])
        # self.ff = [FeedForwardBNN() for _ in range(n_species)]

    @name_count
    def forward(self, x, y=None, species_idx=[], batch_idx=[]):
        assert len(species_idx) > 0, "Species index list is empty"
        assert len(batch_idx) > 0, "Batch index list is empty"

        atomic_energy_contributions = torch.zeros_like(species_idx, device=device, dtype=torch.float32)
        for i in range(len(self.ff)):
            atomic_energy_contributions[species_idx == i] = self.ff[i](x[species_idx == i]).squeeze()

        cohesive_energies = torch.zeros(batch_idx[-1]+1, device=device)
        cohesive_energies.scatter_add_(0, batch_idx, atomic_energy_contributions)

        sigma = pyro.sample('sigma', dist.Uniform(torch.Tensor([0.0]).cuda(), torch.Tensor([0.1]).cuda()))
        with pyro.plate('data', batch_idx[-1]+1, use_cuda=torch.cuda.is_available()):
            obs = pyro.sample('obs', dist.Normal(cohesive_energies, sigma), obs=y)
        return cohesive_energies  # return obs here?


model = BNN()
guide = AutoDiagonalNormal(model)
if torch.cuda.is_available():
    model = model.cuda()
    guide = guide.cuda()

dataset = InMemoryPANNA(standardize_x=True, standardize_y=True)
train_size = int(len(dataset) * 0.9)
train_set, test_set = torch.utils.data.random_split(dataset, lengths=(train_size, len(dataset)-train_size))
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=custom_collate, pin_memory=True, shuffle=True)

optimizer = pyro.optim.Adam({'lr': LR})
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_STEP_MILESTONES, gamma=GAMMA)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))

epochs = 5000

if __name__ == "__main__":
    if RECORD:
        wandb.watch(model, log_freq=1, log='all')
    model.train()

    for e in range(epochs):
        with tqdm.tqdm(train_dataloader) as pbar:
            for batch in pbar:
                x, species_idx, y, batch_idx = batch
                if torch.cuda.is_available():
                    x, species_idx, y, batch_idx = x.cuda(), species_idx.cuda(), y.cuda(), batch_idx.cuda()
                loss = svi.step(x, y, species_idx, batch_idx)
                pbar.set_description('Epoch %i / %i; Loss: %.3f' % (e, epochs, loss/(batch_idx[-1]+1)))
            if RECORD:
                wandb.log({
                    'loss': loss
                })
            # scheduler.step()

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