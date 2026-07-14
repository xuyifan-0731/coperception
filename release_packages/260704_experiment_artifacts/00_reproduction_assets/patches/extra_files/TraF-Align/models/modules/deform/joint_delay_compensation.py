import torch
from torch import nn


class GatedDelayResidualCompensator(nn.Module):
    """Small residual feature corrector for joint delay-compensation training."""

    def __init__(self, channels, cfg):
        super().__init__()
        comp_cfg = cfg["model"].get("delay_compensation", {})
        hidden = int(comp_cfg.get("hidden_channels", max(32, channels // 2)))
        self.max_delay = float(comp_cfg.get("max_delay_frames", 9.0))
        self.residual_scale = float(comp_cfg.get("residual_scale", 0.2))
        self.adaptive_scale = comp_cfg.get("adaptive_scale", None)
        self.bypass_when_ego_delay_zero = bool(comp_cfg.get("bypass_when_ego_delay_zero", False))
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(comp_cfg.get("gate_init", -4.0))))

    def _delay_tensor(self, data_dict, batch_size, device, dtype):
        delays = data_dict["time_delays"]
        ego, cav = [], []
        for item in delays:
            if torch.is_tensor(item):
                vals = item.detach().flatten().tolist()
            else:
                vals = list(item)
            ego.append(float(vals[0]) if len(vals) > 0 else 0.0)
            cav.append(float(vals[1]) if len(vals) > 1 else 0.0)
        vals = ego[:batch_size] + cav[:batch_size]
        if len(vals) < batch_size * 2:
            vals += [0.0] * (batch_size * 2 - len(vals))
        return torch.as_tensor(vals, device=device, dtype=dtype).view(-1, 1, 1, 1)

    def forward(self, x, data_dict):
        # x order after regroup is [all ego agents, all infrastructure agents].
        batch_size = int(data_dict["record_len"].shape[0])
        delay = self._delay_tensor(data_dict, batch_size, x.device, x.dtype)
        if self.bypass_when_ego_delay_zero and torch.all(delay[:batch_size] <= 0):
            return x
        delay_norm = torch.clamp(delay / max(self.max_delay, 1.0), min=0.0, max=1.0)
        active = (delay > 0).to(dtype=x.dtype)
        if active.sum().item() == 0:
            return x
        delay_map = delay_norm.expand(-1, 1, x.shape[-2], x.shape[-1])
        residual = self.net(torch.cat([x, delay_map], dim=1))
        gate = torch.sigmoid(self.gate_logit) * self.residual_scale
        if self.adaptive_scale is not None:
            start = float(self.adaptive_scale.get("start_delay", 2.0))
            end = float(self.adaptive_scale.get("end_delay", self.max_delay))
            min_scale = float(self.adaptive_scale.get("min_scale", 0.1))
            denom = max(end - start, 1e-6)
            factor = 1.0 - (1.0 - min_scale) * torch.clamp((delay - start) / denom, 0.0, 1.0)
            gate = gate * factor
        return x + residual * active * gate
