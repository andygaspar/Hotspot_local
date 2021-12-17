import numpy as np
import torch
from torch import nn, optim


class FFNetworkImitation:
    device: torch.device
    inputDimension: int
    hidden: int
    network: torch.nn.Sequential

    def __init__(self, input_dim: int, n_flights, bids_size, device):
        self.device = device
        self.loss = 0
        self.error = 0
        self.inputDimension = input_dim
        self.numFlights = n_flights
        self.bidsSize = bids_size
        self.outputDimension = (bids_size + 1)
        self.hidden = 10
        self.network_f = nn.Sequential(
            nn.Linear(30, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        ).to(self.device)
        self.network_1 = nn.Sequential(
            nn.Linear(self.inputDimension -30, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
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
        self.optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-3)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)

        self.mu = []
        self.sigma = []
        self.actions = []
    #
    # def sample_action(self, input_vect: torch.tensor) -> int:
    #     return torch.argmax(torch.flatten(input_vect)).item()

    def forward(self, input_vect):
        f = input_vect[:, :30]
        schedule = input_vect[:, 30:]
        f = self.network_f(f)
        s = self.network_1(schedule)

        x = torch.concat((f,s), dim=-1)

        mu = self.network_mean(x)
        sigma = torch.abs(self.network_variance(x))
        return mu, sigma

    def get_bids(self, input_vect: torch.tensor):
        with torch.no_grad():
            mu, sigma = self.forward(input_vect)
            bids = torch.normal(mean=mu, std=sigma)
            return bids

    def update_weights(self, batch):
        NM = torch.distributions.MultivariateNormal

        states, actions, rewards = batch

        loss = 0
        error = 0
        self.mu, self.sigma, self.actions = [], [], []
        for i in range(len(states)):
            mu, sigma = self.forward(states[i])
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.actions.append(actions[i])
            error += torch.mean(torch.abs(mu - sigma))

            loss += - 1/len(states) * NM(mu.flatten(), torch.eye(len(sigma.flatten())).to(self.device)*sigma.flatten()).log_prob(actions[i].flatten())

        self.error = error/len(states)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        # print("loss", loss.item())
        #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

    def print_params(self):
        for i, action in enumerate(self.actions):
            print("actions", action)
            print("mu", self.mu[i])
            print("si", self.sigma[i])

    def take_weights(self, model_network):
        self.network.load_state_dict(model_network.network.state_dict())

    def load_weights(self, file):
        self.network.load_state_dict(torch.load(file))

    def save_weights(self, filename: str):
        torch.save(self.network.state_dict(), filename + '.pt')

