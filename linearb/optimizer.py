import torch

class OrthogonalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, num_flips=1):
        defaults = dict(lr=lr, momentum=momentum, num_flips=num_flips)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if not getattr(p, 'is_quantized_basis', False): # gradient descent on biases and non linearb layers
                    p.data -= group['lr'] * p.grad.data
                    continue
                # print('psassed the threshold')
                num_bases, nxn = p.binary_coeffs.shape
                n = int(nxn**0.5)
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.binary_coeffs)
                m = self.state[p]["momentum_buffer"]
                m = group["momentum"] * m + (1 - group["momentum"]) * p.grad
                self.state[p]["momentum_buffer"] = m
                i = (m * p).view(-1).argmax()
                p.view(-1)[i] *= -1
                # for i in range(num_bases):
                #     c = p.binary_coeffs[i].view(n, n)
                #     m_rotated = p.U_L[i].T @ m @ p.U_R[i].T
                #     mb = self.hb_transform(m_rotated.view(1, -1)).view(n, n)
                #     scores = mb * c
                #     _, min_wi = torch.topk(scores.view(-1), min(group["num_flips"], n * n), largest=False)
                #     for min_i in min_wi:
                #         flipr = min_i // n
                #         flipc = min_i % n
                #         if scores[flipr, flipc] < 0:
                #             c[flipr, flipc] *= -1
                #     p.binary_coeffs[i] = c.view(-1)
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
        return x.reshape(m, 2**k, 2**k) / (2**k)