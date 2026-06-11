import torch

class TrellisOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, num_flips=1):
        defaults = dict(lr=lr, momentum=momentum, num_flips=num_flips)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}
                
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if not getattr(p, 'is_trellis', False): # gradient descent on biases and non lineart layers
                    p.data -= group['lr'] * p.grad.data
                    continue
                W = getattr(p, 'W', None)
                if W is None or W.grad is None:
                    continue
                W_grad = W.grad.data
                states = p.states
                block_size = p.block_size
                num_blocks = p.num_blocks
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(W_grad)
                m = self.state[p]["momentum_buffer"]
                m = group["momentum"] * m + (1 - group["momentum"]) * W_grad
                self.state[p]["momentum_buffer"] = m
                m = m.view(p.out_params // block_size, block_size, p.in_params // block_size, block_size)
                m = m.permute(0, 2, 1, 3).reshape(num_blocks, block_size, block_size)
                curr_state = "00"
                curr_path = []
                for bit in p.data.detach().cpu().numpy().astype(int):
                    curr_path.append(curr_state)
                    next_state, _ = states[(curr_state, int(bit.item()))]
                    curr_state = next_state
                scores = torch.zeros(num_blocks, device=p.device)
                alts = torch.zeros(num_blocks, device=p.device)
                for i in range(num_blocks):
                    alts[i] = 1 if int(p.data[i].item()) == 0 else 0
                    prev_state = curr_path[i]
                    _, current_mat = states[(prev_state, int(p.data[i].item()))]
                    _, flipped_mat = states[(prev_state, alts[i])]
                    W_change = flipped_mat - current_mat
                    scores[i] = -torch.sum(W_change * m[i])
                top_scores, top_indices = torch.topk(scores, min(group["num_flips"], num_blocks))
                for score, idx in zip(top_scores, top_indices):
                    if score > 0:
                        p.data[idx] = alts[idx]
        return loss
