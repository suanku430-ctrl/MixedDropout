import torch
import torch.nn as nn
import torch.nn.functional as F


class MixDropout(nn.Module):
    def __init__(self, in_features, out_features, p_init=0.5, rho_init=0.5, use_rho_init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.use_rho_init = use_rho_init
        if use_rho_init:
            self.register_buffer('rho', torch.full((in_features,), rho_init))
            self.register_buffer('p', torch.zeros(in_features))
        else:
            self.register_buffer('p', torch.full((in_features,), p_init))
            self.register_buffer('rho', torch.zeros((in_features,)))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_detached = x.detach()
                S = torch.norm(x_detached, p=2, dim=0)
                S_max = torch.max(S)
                keep = S / (S_max + 1e-8)
                eps = 1e-8
                if self.use_rho_init:
                    self.p.data = torch.clamp(1 - keep / ((1 - self.rho) + eps), 0, 1)
                else:
                    self.rho.data = torch.clamp(1 - keep / ((1 - self.p) + eps), 0, 1)

            mask_weight = torch.bernoulli((1 - self.rho).unsqueeze(0).expand(self.out_features, -1)).to(x.device)
            masked_weight = self.weight * mask_weight
            mask_input = (torch.rand_like(x) < (1 - self.p)).to(x.dtype) / (1 - self.p + eps)
            x = x * mask_input
            return F.linear(x, masked_weight, self.bias)
        else:
            expected_weight = self.weight * (1 - self.rho).view(1, -1)
            return F.linear(x, expected_weight, self.bias)


class RadDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            x_detached = x.detach()
            S = torch.sqrt(torch.sum(x_detached ** 2, dim=0))
            S_max = torch.max(S) + 1e-8
            p = torch.clamp(1.0 - S / (S_max + 1e-8), 0.1, 0.9)
            mask_shape = x.shape
            uniform = torch.rand(mask_shape, device=x.device)
            keep_prob = (1 - p).unsqueeze(0)
            mask = torch.where(
                uniform < keep_prob,
                torch.tensor(1.0 / keep_prob, device=x.device),
                torch.tensor(0.0, device=x.device)
            )
            return x * mask
        else:
            return x


class DropConnect(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
       
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.p = p
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training and self.p > 0.0:
            keep_prob = 1 - self.p
            mask = torch.bernoulli(torch.full(self.weight.shape, keep_prob, device=x.device))
            masked_weight = self.weight * mask * (1 / keep_prob)
            return F.linear(x, masked_weight, self.bias)
        return F.linear(x, self.weight, self.bias)

class AdaptiveDropConnect(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.rho = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            alpha = torch.norm(x, p=2, dim=0)
            with torch.no_grad():
                self.rho.copy_(1.0 - alpha / (alpha.max() + 1e-8))
            rho_matrix = self.rho.unsqueeze(0).expand(self.out_features, -1)
            mask = torch.bernoulli(1 - rho_matrix).to(x.device)
            return F.linear(x, self.weight * mask, self.bias)
        else:
            return F.linear(x, self.weight * (1 - self.rho).view(1, -1), self.bias)