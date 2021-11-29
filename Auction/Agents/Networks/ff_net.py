import numpy as np
import torch
from torch import nn, optim


class FFNetwork:
    device: torch.device
    inputDimension: int
    hidden: int
    network: torch.nn.Sequential

    def __init__(self, input_dim: int, n_flights, bids_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = 0
        self.inputDimension = input_dim
        self.numFlights = n_flights
        self.bidsSize = bids_size
        self.outputDimension = n_flights * bids_size
        self.hidden = 10
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, 30),
            nn.LeakyReLU(),
            nn.Linear(30, self.outputDimension)
            # nn.LeakyReLU(),
            # nn.Linear(self.inputDimension, self.inputDimension),
            # nn.LeakyReLU(),
            # nn.Linear(self.inputDimension, self.inputDimension),
            # nn.LeakyReLU(),
            # nn.Linear(2 * self.inputDimension, 2 * self.inputDimension),
            # nn.LeakyReLU(),
            # nn.Linear(2 * self.inputDimension,  self.inputDimension),
        )
        self.network.to(self.device)
        torch.cuda.current_device()
        print(torch.cuda.is_available())
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-5, weight_decay=1e-5)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)
    #
    # def sample_action(self, input_vect: torch.tensor) -> int:
    #     return torch.argmax(torch.flatten(input_vect)).item()

    def get_bids(self, input_vect: np.array):
        X = torch.from_numpy(input_vect).to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32)
        with torch.no_grad():
            bids = torch.flatten(self.network(X)).to(self.device)
            return bids.cpu().numpy().reshape(self.numFlights, self.bidsSize)


    def update_weights(self, batch: tuple, gamma: float, target_network):
        criterion = torch.nn.MSELoss()

        states, actions, nextStates, rewards, dones = batch

        # if sum(dones) > 0:
        #    pass

        X = torch.tensor([el.tolist() for el in states]).to(self.device).float().reshape(-1, self.inputDimension)
        X_next = torch.tensor([el.tolist() for el in nextStates]).to(self.device)\
            .reshape(-1, self.inputDimension)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=int).to(self.device)
        for i in range(5):
            curr_Q = self.network(X).gather(1, actions.unsqueeze(1)).to(self.device).squeeze(1)

            with torch.no_grad():
                next_Q = target_network.network(X_next).to(self.device)
                max_next_Q = torch.max(next_Q, 1)[0]
                expected_Q = (rewards + (1 - dones) * gamma * max_next_Q).to(self.device)

            loss = criterion(curr_Q, expected_Q.detach())
            self.loss = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

    def take_weights(self, model_network):
        self.network.load_state_dict(model_network.network.state_dict())

    def load_weights(self, file):
        self.network.load_state_dict(torch.load(file))

    def save_weights(self, filename: str):
        torch.save(self.network.state_dict(), filename + '.pt')