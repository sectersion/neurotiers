import torch
from torch import nn


def _spike(membrane, threshold):
    """Hard spike forward pass with a smooth surrogate backward pass."""
    soft = torch.sigmoid(10 * (membrane - threshold))
    hard = (membrane >= threshold).to(membrane.dtype)
    return hard + soft - soft.detach()


class IF(nn.Module):
    """Integrate-and-fire neuron with a hard reset."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        membrane = torch.zeros_like(x[0])
        spikes = []
        for current in x:
            membrane = membrane + current
            spike = _spike(membrane, self.threshold)
            membrane = membrane * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes)


class LIF(IF):
    """Leaky integrate-and-fire neuron."""
    def __init__(self, threshold=1.0, decay=0.8):
        super().__init__(threshold)
        self.decay = decay

    def forward(self, x):
        membrane = torch.zeros_like(x[0])
        spikes = []
        for current in x:
            membrane = self.decay * membrane + current
            spike = _spike(membrane, self.threshold)
            membrane = membrane * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes)


class AdaptiveLIF(LIF):
    """LIF neuron whose threshold increases after each spike."""
    def __init__(self, threshold=1.0, decay=0.8, adaptation=0.2):
        super().__init__(threshold, decay)
        self.adaptation = adaptation

    def forward(self, x):
        membrane = torch.zeros_like(x[0])
        extra = torch.zeros_like(membrane)
        spikes = []
        for current in x:
            membrane = self.decay * membrane + current
            spike = _spike(membrane, self.threshold + extra)
            membrane = membrane * (1 - spike)
            extra = self.decay * extra + self.adaptation * spike
            spikes.append(spike)
        return torch.stack(spikes)


class Izhikevich(nn.Module):
    """Izhikevich spiking neuron using the regular-spiking parameters."""
    def __init__(self, threshold=30.0, dt=1.0, a=0.02, b=0.2,
                 c=-65.0, d=8.0, gain=200.0):
        super().__init__()
        self.threshold, self.dt = threshold, dt
        self.a, self.b, self.c, self.d = a, b, c, d
        self.gain = gain

    def forward(self, x):
        voltage = torch.full_like(x[0], self.c)
        recovery = self.b * voltage
        spikes = []
        for current in x:
            dv = 0.04 * voltage.square() + 5 * voltage + 140 - recovery + self.gain * current
            voltage = voltage + self.dt * dv
            recovery = recovery + self.dt * self.a * (self.b * voltage - recovery)
            spike = _spike(voltage, self.threshold)
            hard = (voltage >= self.threshold).to(voltage.dtype)
            voltage = torch.where(hard.bool(), torch.as_tensor(self.c, device=x.device, dtype=x.dtype), voltage)
            recovery = recovery + self.d * hard
            spikes.append(spike)
        return torch.stack(spikes)


class HodgkinHuxley(nn.Module):
    """Hodgkin-Huxley neuron integrated with an explicit Euler step.

    Voltages are relative to resting potential (-65 mV). The reset after a
    spike is set just below threshold to model refractoriness.
    """
    def __init__(self, threshold=-20.0, dt=0.01, gain=5.0, v_reset=-25.0):
        super().__init__()
        self.threshold, self.dt = threshold, dt
        self.gain = gain
        self.v_reset = v_reset

    def forward(self, x):
        voltage = torch.zeros_like(x[0])
        m = torch.full_like(voltage, 0.0529)
        h = torch.full_like(voltage, 0.596)
        n = torch.full_like(voltage, 0.317)
        spikes = []
        for current in x:
            absolute_v = voltage - 65.0
            am = 0.1 * (25 - absolute_v) / (torch.exp((25 - absolute_v) / 10) - 1 + 1e-8)
            bm = 4 * torch.exp(-absolute_v / 18)
            ah = 0.07 * torch.exp(-absolute_v / 20)
            bh = 1 / (torch.exp((30 - absolute_v) / 10) + 1)
            an = 0.01 * (10 - absolute_v) / (torch.exp((10 - absolute_v) / 10) - 1 + 1e-8)
            bn = 0.125 * torch.exp(-absolute_v / 80)
            m = (m + self.dt * (am * (1 - m) - bm * m)).clamp(0, 1)
            h = (h + self.dt * (ah * (1 - h) - bh * h)).clamp(0, 1)
            n = (n + self.dt * (an * (1 - n) - bn * n)).clamp(0, 1)
            ionic = (120 * m**3 * h * (voltage - 50) +
                     36 * n**4 * (voltage + 77) + 0.3 * (voltage + 54.4))
            voltage = voltage + self.dt * (self.gain * current - ionic)
            voltage = voltage.clamp(-100, 100)
            spike = _spike(voltage, self.threshold)
            voltage = torch.where(spike > 0.5,
                                 torch.as_tensor(self.v_reset, device=voltage.device, dtype=voltage.dtype),
                                 voltage)
            spikes.append(spike)
        return torch.stack(spikes)
