import numpy as np
import torch

class OrthogonalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, num_bits=1, num_flips=1):
        defaults = dict(lr=lr, momentum=momentum, num_bits=num_bits, num_flips=num_flips)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if len(p.shape) < 2: # skip biases
                    continue
                self.state[p] = {}
                self.state[p]["basis_rots"] = []
                self.state[p]["binary_coeffs"] = []
                row, col = p.shape
                n = max(2, int(2 ** np.ceil(np.log2(max(row, col))))) # need to find what dim we want for basis since not always power of 2 or square
                self.state[p]["momentum_buffer"] = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                sum_w_dec_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                for i in range(group["num_bits"]):
                    if i == 0:
                        self.state[p]["basis_rots"].append((torch.eye(n, device=p.device), torch.eye(n, device=p.device)))
                    else:
                        l = torch.randn(n, n, device=p.device)
                        U_l, _ = torch.linalg.qr(l)
                        r = torch.randn(n, n, device=p.device)
                        U_r, _ = torch.linalg.qr(r)
                        self.state[p]["basis_rots"].append((U_l, U_r))
                    w_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                    w_padded[:row, :col] = p.data
                    w_rotated = self.state[p]["basis_rots"][-1][0].T @ w_padded @ self.state[p]["basis_rots"][-1][1] 
                    w_flat = w_rotated.view(1, -1)
                    c_raw = self.hb_transform(w_flat).view(n, n)
                    c = torch.sign(c_raw)
                    c[c == 0] = 1
                    self.state[p]["binary_coeffs"].append(c)
                    c_dec = self.hb_transform(c.view(1, -1)).view(n, n) / (n * n)
                    w_dec_padded = self.state[p]["basis_rots"][-1][0] @ c_dec @ self.state[p]["basis_rots"][-1][1].T
                    sum_w_dec_padded += w_dec_padded
                sum_w_dec = (sum_w_dec_padded / group["num_bits"])[:row, :col]
                with torch.no_grad():
                    p.data.copy_(sum_w_dec)
    
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
                m = self.state[p]["momentum_buffer"]
                row, col = p.shape
                n = m.shape[0]
                grad_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                grad_padded[:row, :col] = p.grad.data
                m = group["momentum"] * m + (1 - group["momentum"]) * grad_padded
                m[row:, :] = 0
                m[:, col:] = 0
                self.state[p]["momentum_buffer"] = m
                sum_w_dec_padded = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                for i in range(group["num_bits"]):
                    c = self.state[p]["binary_coeffs"][i]
                    m_rotated = self.state[p]["basis_rots"][i][0].T @ m @ self.state[p]["basis_rots"][i][1]
                    mb = self.hb_transform(m_rotated.view(1, -1)).view(n, n)
                    scores = mb * c
                    # assuming we update 1 bit per step, differs by basis
                    _, min_wi = torch.topk(scores.view(-1), min(group["num_flips"], n * n), largest=False)
                    for min_i in min_wi:
                        flipr = min_i // n
                        flipc = min_i % n
                        if scores[flipr, flipc] < 0:
                            c[flipr, flipc] *= -1
                    self.state[p]["binary_coeffs"][i] = c
                    c_dec = self.hb_transform(c.view(1, -1)).view(n, n) / (n * n)
                    w_dec_padded = self.state[p]["basis_rots"][i][0] @ c_dec @ self.state[p]["basis_rots"][i][1].T
                    sum_w_dec_padded += w_dec_padded
                sum_w_dec = (sum_w_dec_padded / group["num_bits"])[:row, :col]
                with torch.no_grad():
                    p.data.copy_(sum_w_dec)
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