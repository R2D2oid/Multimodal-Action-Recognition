import torch
import torch.nn as nn
from layers.AttentionLayer import AttentionLayer
# AutoEncoder model with attention 

class AEwithAttention(nn.Module):
    def __init__(self, n_feat, T, n_filt):
        super(AEwithAttention, self).__init__()
        
        z_dim = n_feat*n_filt
        
        self.attn_enc_ = AttentionLayer(n_feat, T, n_filt, transposeit = False)
        self.attn_dec_ = AttentionLayer(n_feat, T, n_filt, transposeit = True)
        
        # defining Encoder network with an attention layer (with N filters) followed by 4 fully connected layers
        self.encoder_ = nn.Sequential(
            # attn_enc is to be places before all the following layers
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim)
         )
        
        # defining Decoder network with 4 fully connected layers followed by a transposed attention layer (with N filters)
        self.decoder_ = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            # attn_dec is to be places after all the above layers
        )
         

    def encoder(self, x):
        x = self.attn_enc_(x)
        x = self.encoder_(x)
        # add L2 normalization to get L2_norm(embedding) = 1
        return x
    
    def decoder(self, x):
        x = self.decoder_(x)
        x = self.attn_dec_(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
