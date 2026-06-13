import torch
import torch.nn as nn

class LinearSB(nn.Module):
    def __init__(self, in_params, out_params, num_bases):
        super().__init__()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_params = in_params
        self.out_params = out_params
        self.num_bases = num_bases
        self.n = max(2, int(2 ** torch.ceil(torch.log2(torch.tensor(max(in_params, out_params))))))
        binary_bits = torch.sign(torch.randn(num_bases, self.n * self.n, device=current_device))
        self.binary_coeffs = torch.nn.Parameter(binary_bits) # (b, n^2)
        self.binary_coeffs.is_quantized_basis = True
        self.binary_coeffs.binary_coeffs = self.binary_coeffs
        self.binary_coeffs.true_shape = (out_params, in_params)
        self.bias = torch.nn.Parameter(torch.zeros(out_params, device=current_device))
        self.bias.is_quantized_basis = False
        self.scale = 1 / ((self.in_params * self.num_bases) ** 0.5)

        U_Ls = []
        U_Rs = []
        for i in range(num_bases):
            if i == 0:
                U_Ls.append(torch.eye(self.n, device=current_device))
                U_Rs.append(torch.eye(self.n, device=current_device))
            else:
                l = torch.randn(self.n, self.n, device=current_device)
                U_l, _ = torch.linalg.qr(l)
                r = torch.randn(self.n, self.n, device=current_device)
                U_r, _ = torch.linalg.qr(r)
                U_Ls.append(U_l)
                U_Rs.append(U_r)
        self.register_buffer("U_L", torch.stack(U_Ls))
        self.register_buffer("U_R", torch.stack(U_Rs))
        self.binary_coeffs.U_L = self.U_L
        self.binary_coeffs.U_R = self.U_R
    
    def weight(self):
        c = self.binary_coeffs.view(self.num_bases, self.n, self.n)
        W = self.U_L @ c @ self.U_R
        W = torch.sum(W, dim=0)[:self.out_params, :self.in_params]
        return W * self.scale
    
    def forward(self, x):
        W = self.weight()
        return x @ W.T + self.bias