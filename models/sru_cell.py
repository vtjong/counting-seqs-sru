import math
import sys
import torch
import torch.nn.init as init
import torch.nn as nn

class SRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.p = dropout

        self.init_parameters_()
        self.reset_parameters()

    def init_parameters_(self):
        self.W = nn.Parameter(torch.Tensor(self.input_size, 4*self.output_size))
        self.V = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.b = nn.Parameter(torch.zeros(2 * self.output_size))

    def reset_parameters(self):
        sigma = lambda d=1: (3.0 / d)**0.5
        d = self.W.size(0)
        init.uniform_(self.W, -sigma(d), sigma(d))
        init.uniform_(self.V, -sigma(), sigma())
    
    def dropout(self, B, d, x):
        size = (B, self.output_size)
        p = self.p
        dropout_mask = torch.empty(size).bernoulli_(1 - p).div_(1 - p) \
                            if self.training and (self.p > 0) else None
        dropout_mask = dropout_mask.view(B, d) if dropout_mask is not None else None
        return x if dropout_mask is None else dropout_mask[:, :] * x

    def forward(self, input, c0=None):
        batch_size = input.size(-2)
        
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size)

        U = input.contiguous().view(-1, self.input_size).mm(self.W)

        def forward_(self, U, V, x, c0):
            d = self.hidden_size
            L = x.size(0)
            B = x.size(-2)
            U = U.contiguous().view(L, B, d, -1)
            
            v_f, v_r = V.view(2, d)
            v_f, v_r = v_f.expand(B, d), v_r.expand(B, d) 
            b_f, b_r = self.b.view(2, d)

            c_prev = c0
            h = torch.Tensor(L, B, d)

            for l in range(L):
                f_t = torch.sigmoid((U[l, ..., 1] + c_prev*v_f) + b_f)
                r_t = torch.sigmoid((U[l, ..., 2] + c_prev*v_r) + b_r)
                c_t = U[l, ..., 0] + (c_prev - U[l, ..., 0]) * f_t
                c_prev = c_t    
                h_t = self.dropout(B, d, (c_t.tanh() - U[l, ..., 3]) * r_t)
                h_t += U[l, ..., 3]
                h[l, ...] = h_t
            return h, c_t

        return forward_(self, U, self.V, input, c0)