import torch
import torch.nn as nn
import math

# attention layer composed of N gaussian filters
class AttentionLayer(nn.Module):
    def __init__(self, n_feat, T, n_filt, transposeit = False):
        super(AttentionLayer, self).__init__()
        self.T = T
        self.n_filt = n_filt
        self.n_feat = n_feat
        self.transposeit = transposeit
        
        self.mu = torch.rand((n_filt, 1), requires_grad = True)
        self.sig = torch.rand((n_filt, 1), requires_grad = True)
            
        self.attn_param = nn.ParameterList([nn.Parameter(self.mu), nn.Parameter(self.sig)])        
        
    def forward(self, x):
        # depending on whether the filter is located in the encoder or decoder we need to transpose in order to conform dimensions for matmul
        if self.transposeit:
            x = x.reshape(self.n_feat, self.n_filt)
        else:
            x = x.reshape(self.n_feat, self.T)
            
        # get the attention matrix composed of N filters
        t = time_steps(self.n_filt, 1, self.T, self.T)
        #attentions = gaussian(self.mu, self.sig, t)
        attentions = gaussian(self.attn_param[0], self.attn_param[1], t)
        
        attentions = attentions.t()
        
        if self.transposeit:
            self.attended = torch.matmul(x, attentions.t())
        else:        
            self.attended = torch.matmul(x, attentions)
            
        self.attended = self.attended.reshape((1, -1))
        
        return self.attended

# define constants
EPS = 1e-10
SQRT_PI = math.sqrt(math.pi)

# define gaussians for attention filter
# derivatives will be calculated wrt mu, sigma
def gaussian(mu, sig, x):
    z = x - mu
    z2 = z.pow(2)
    sig = sig + EPS  # EPS to avoid divide by 0
    return  torch.exp(-z2) / (2 * sig * SQRT_PI)

# define full attention matrix composed of N gaussians
def time_steps(n, T_start, T_end, T_steps, requires_grad = False):
    m = torch.ones((n, 1)) * torch.linspace(T_start, T_end, T_steps)
    m.requires_grad = requires_grad
    return m
