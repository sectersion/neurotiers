import torch
from torch import nn

from .neurons import AdaptiveLIF, HodgkinHuxley, IF, Izhikevich, LIF


class SNNClassifier(nn.Module):
    def __init__(self, features=8, hidden=16, classes=2, neuron="lif", steps=12):
        super().__init__()
        neurons = {"if": IF, "lif": LIF, "adaptive-lif": AdaptiveLIF,
                   "izhikevich": Izhikevich, "hodgkin-huxley": HodgkinHuxley}
        if neuron not in neurons:
            raise ValueError(f"unknown neuron: {neuron}")
        self.steps = steps
        self.encoder = nn.Linear(features, hidden)
        self.neuron = neurons[neuron]()
        self.readout = nn.Linear(hidden, classes)

    def forward(self, rates):
        currents = self.encoder(rates).relu().unsqueeze(0).expand(self.steps, -1, -1)
        spikes = self.neuron(currents)
        return self.readout(spikes.mean(0)), spikes
