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
        self.outputDimension = n_flights * (bids_size + 1)
        self.hidden = 10
        self.network_1 = nn.Sequential(
            nn.Linear(self.inputDimension, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30)
        ).to(self.device)
        self.network_mean = nn.Sequential(
            nn.Linear(30, self.outputDimension)
        ).to(self.device)
        self.network_variance = nn.Sequential(
            nn.Linear(30, self.outputDimension)
        ).to(self.device)
        torch.cuda.current_device()
        print(torch.cuda.is_available())
        params = list(self.network_1.parameters()) + list(self.network_mean.parameters()) + list(self.network_variance.parameters())
        self.optimizer = optim.Adam(params, lr=1e-5, weight_decay=1e-5)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)
    #
    # def sample_action(self, input_vect: torch.tensor) -> int:
    #     return torch.argmax(torch.flatten(input_vect)).item()

    def forward(self, input_vect):
        x = self.network_1(input_vect)
        mu = self.network_mean(x)
        sigma = torch.abs(self.network_variance(x))
        return mu, sigma

    def get_bids(self, input_vect: torch.tensor):
        X = input_vect.reshape(1, self.inputDimension)
        with torch.no_grad():
            mu, sigma = self.forward(X)
            bids = torch.normal(mean=mu, std=sigma)
            return bids

    def update_weights(self, batch):
        NM = torch.distributions.MultivariateNormal

        states, actions, rewards = batch

        # if sum(dones) > 0:
        #    pass
        loss = 0
        for i in range(len(states)):
            mu, sigma = self.forward(states[i])
            loss += rewards[i] * NM(mu, torch.eye(len(sigma)).to(self.device)*sigma).log_prob(actions[i])

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

