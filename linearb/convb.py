import torch
import torch.nn as nn

class ConvB(nn.Module):
    def __init__(self, in_params, out_params, kernel_size, num_bases, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_params = in_params
        self.out_params = out_params
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_bases = num_bases
        self.dim1, self.dim2 = out_params, in_params * self.kernel_size[0] * self.kernel_size[1]
        self.n = max(2, int(2 ** torch.ceil(torch.log2(torch.tensor(max(self.dim1, self.dim2))))))
        binary_bits = torch.sign(torch.randn(num_bases, self.n * self.n, device=current_device)) # (b, n^2)
        self.binary_coeffs = torch.nn.Parameter(binary_bits, requires_grad=False) # (b, n^2)
        self.binary_coeffs.is_quantized_basis = True
        self.binary_coeffs.binary_coeffs = self.binary_coeffs
        self.binary_coeffs.true_shape = (out_params, in_params)
        self.bias = torch.nn.Parameter(torch.zeros(out_params, device=current_device))
        self.bias.is_quantized_basis = False

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
    
    def forward(self, x):
        W = self.hb_transform(self.binary_coeffs)
        W = self.U_L @ W @ self.U_R
        W = torch.sum(W, dim=0)[:self.dim1, :self.dim2]
        W = W.view(self.out_params, self.in_params, self.kernel_size[0], self.kernel_size[1])
        return nn.functional.conv2d(x, W, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    
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