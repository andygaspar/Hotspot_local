import numpy as np
import torch
from torch import nn, optim


class FFNetwork:
    device: torch.device
    inputDimension: int
    hidden: int
    network: torch.nn.Sequential

    def __init__(self, input_dim: int, n_flights, bids_size, device):
        self.device = device
        self.loss = 0
        self.inputDimension = input_dim
        self.numFlights = n_flights
        self.bidsSize = bids_size
        self.outputDimension = (bids_size + 1)
        self.hidden = 10
        self.network_1 = nn.Sequential(
            nn.Linear(self.inputDimension, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        self.network_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.outputDimension)
        ).to(self.device)
        self.network_variance = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.outputDimension)
        ).to(self.device)
        torch.cuda.current_device()
        print(torch.cuda.is_available())
        params = list(self.network_1.parameters()) + list(self.network_mean.parameters()) + list(self.network_variance.parameters())
        self.optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-2)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)
    #
    # def sample_action(self, input_vect: torch.tensor) -> int:
    #     return torch.argmax(torch.flatten(input_vect)).item()

    def forward(self, input_vect):
        x = self.network_1(input_vect)
        mu = self.network_mean(x)
        sigma = torch.abs(self.network_variance(x) + 20)
        return mu, sigma

    def get_bids(self, input_vect: torch.tensor):
        with torch.no_grad():
            mu, sigma = self.forward(input_vect)
            bids = torch.normal(mean=mu, std=sigma)
            return bids

    def update_weights(self, batch, print_params=False):
        NM = torch.distributions.MultivariateNormal

        states, actions, rewards = batch

        loss = 0
        for i in range(len(states)):
            s = states[i]
            r = rewards[i]
            a = actions[i]
            mu, sigma = self.forward(states[i])
            if print_params:
                print(rewards[i])
                print("mu", mu)
                print("si", sigma)
                print("actions", actions[i])
                print("log", NM(mu.flatten(), torch.eye(len(sigma.flatten())).to(self.device)*sigma.flatten()).log_prob(actions[i].flatten()),"\n\n")
            loss += - 1/len(states) * (3000 - rewards[i]) * NM(mu.flatten(), torch.eye(len(sigma.flatten())).to(self.device)*sigma.flatten()).log_prob(actions[i].flatten())

        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        print("loss", loss.item())
        #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

    def take_weights(self, model_network):
        self.network.load_state_dict(model_network.network.state_dict())

    def load_weights(self, file):
        self.network.load_state_dict(torch.load(file))

    def save_weights(self, filename: str):
        torch.save(self.network.state_dict(), filename + '.pt')


#
# import torch
# import numpy as np
# import scipy.stats as ss
#
# import matplotlib.pyplot as plt
#
# NM = torch.distributions.MultivariateNormal
# n = 20
# sigma_ = 0.
# mu = torch.zeros(n)
# sigma = torch.ones(n)*sigma_
# actions = torch.zeros(n)
# print(NM(mu.flatten(), torch.diag(sigma.flatten())).log_prob(actions))
#
# ff = ss.multivariate_normal(np.zeros(n), np.diag(np.ones(n)*sigma_))
#
# print(ff.logpdf(np.zeros(n)))