import torch

class TrellisOptimizer(torch.optim.Optimizer):
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
                if not getattr(p, 'is_trellis', False): # gradient descent on biases
                    p.data -= group['lr'] * p.grad.data
                    continue
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                m = self.state[p]["momentum_buffer"]
                m = group["momentum"] * m + (1 - group["momentum"]) * p.grad
                self.state[p]["momentum_buffer"] = m
                curr_bits = p.view(-1)
                scores = m.view(-1) * (2 * curr_bits - 1) # want to convert to 1/-1 so that scores aren't 0 for bits that are 0 that need to flip
                _, max_i = torch.topk(scores, group["num_flips"])
                for i in max_i:
                    p.view(-1)[i] = 1.0 - p.view(-1)[i]
        return loss
