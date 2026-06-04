import numpy as np
import torch

class OneBitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.basis_mats = {}
        for group in self.param_groups:
            for p in group['params']:
                if len(p.shape) < 2: # skip biases
                    continue
                row, col = p.shape
                n = max(2, int(2 ** np.ceil(np.log2(max(row, col))))) # need to find what dim we want for basis since not always power of 2 or square
                w_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                w_padded[:row, :col] = p.data
                w_flat = w_padded.view(1, -1)
                c_raw = self.hb_transform(w_flat).view(n, n)/(n * n)
                c = torch.sign(c_raw)
                c[c == 0] = 1
                self.state[p]["binary_coeffs"] = c
                self.state[p]["momentum_buffer"] = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                w_dec_padded = self.hb_transform(c.view(1, -1)).view(n, n)
                w_dec = w_dec_padded[:row, :col]
                p.data.copy_(w_dec)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if len(p.shape) < 2: # skip biases
                    continue
                c = self.state[p]["binary_coeffs"]
                m = self.state[p]["momentum_buffer"]
                row, col = p.shape
                n = m.shape[0]
                grad_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                grad_padded[:row, :col] = p.grad.data
                m = group["momentum"] * m + (1 - group["momentum"]) * grad_padded
                m[row:, :] = 0
                m[:, col:] = 0
                self.state[p]["momentum_buffer"] = m
                mb = self.hb_transform(m.view(1, -1)).view(n, n)
                scores = mb * c
                # assuming we only update 1 bit per step
                min_wi = torch.argmin(scores)
                flipr = min_wi // n
                flipc = min_wi % n
                if scores[flipr, flipc] < 0:
                    c[flipr, flipc] *= -1
                self.state[p]["binary_coeffs"] = c
                w_dec_padded = self.hb_transform(c.view(1, -1)).view(n, n)/(n * n)
                w_dec = w_dec_padded[:row, :col]
                p.data.copy_(w_dec)
        return loss
    
    def hb_transform(self, x):
        if len(x.shape) == 1:
            x = x.view(1,-1)
        (m,n) = x.shape
        k = 1
        while 4**k < n:
            k += 1
        assert(4**k == n)
        x = x.reshape((m,) + (4,)*k)
        for i in range(k):
            x = x.sum(1+i,keepdim=True) - 2*x
        x = x.reshape((m,) + (2,2)*k)
        x = x.permute((0,) + tuple(2*i+1 for i in range(k)) + tuple(2*i+2 for i in range(k)))
        return x.reshape(m, 2**k, 2**k)