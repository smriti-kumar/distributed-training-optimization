import torch
import torch.nn as nn

class LinearT(nn.Module):
    def __init__(self, in_params, out_params, block_size):
        super().__init__()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_params = in_params
        self.out_params = out_params
        self.block_size = block_size

        if (in_params * out_params) % (block_size * block_size) != 0:
            raise ValueError("Matrix cant be divided evenly by blocks of inputted size")
        self.num_blocks = in_params * out_params // (block_size * block_size)
        self.mat00, _ = torch.linalg.qr(torch.randn(block_size, block_size, device=current_device))
        self.mat01, _ = torch.linalg.qr(torch.randn(block_size, block_size, device=current_device))
        self.mat01 = self.mat00 @ self.mat01
        self.mat10, _ = torch.linalg.qr(torch.randn(block_size, block_size, device=current_device))
        self.mat10 = self.mat01 @ self.mat10
        self.mat11, _ = torch.linalg.qr(torch.randn(block_size, block_size, device=current_device))
        self.mat11 = self.mat10 @ self.mat11
        self.states = {
            ("00", 0): ("00", self.mat00),
            ("00", 1): ("10", self.mat10),
            ("01", 0): ("00", self.mat00),
            ("01", 1): ("10", self.mat10),
            ("10", 0): ("01", self.mat01),
            ("10", 1): ("11", self.mat11),
            ("11", 0): ("01", self.mat01),
            ("11", 1): ("11", self.mat11),
        }
        binary_bits = torch.randint(0, 2, (self.num_blocks,), dtype=torch.float32, device=current_device) # before it was -1, 1 - do we keep it that way or change to 0, 1
        self.binary_coeffs = torch.nn.Parameter(binary_bits, requires_grad=False)
        self.binary_coeffs.is_trellis = True
        self.bias = torch.nn.Parameter(torch.zeros(out_params, device=current_device))
        self.bias.is_trellis = False
        self.binary_coeffs.states = self.states
        self.binary_coeffs.block_size = self.block_size
        self.binary_coeffs.num_blocks = self.num_blocks
        self.binary_coeffs.out_params = self.out_params
        self.binary_coeffs.in_params = self.in_params

    def forward(self, x):
        curr_state = "00"
        decoded = []
        for bit in self.binary_coeffs.detach().cpu().numpy().astype(int):
            next_state, mat = self.states[(curr_state, int(bit.item()))]
            decoded.append(mat)
            curr_state = next_state
        W = torch.stack(decoded).view(self.out_params // self.block_size, self.in_params // self.block_size, self.block_size, self.block_size)
        W = W.permute(0, 2, 1, 3).reshape(self.out_params, self.in_params)
        W = W.clone().requires_grad_(True)
        self.binary_coeffs.W = W
        return x @ W.T + self.bias