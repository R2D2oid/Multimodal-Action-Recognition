import torch
import torch.nn as nn
import numpy as np

from utilities.preprocessing import preprocess_embeddings
from utilities.utils import load_feats_from 
from layers.AEwithAttention import AEwithAttention

# set params
learning_rate = 0.01
n_epochs = 2
T = 10
n_filt = 4
n_feat = 1024

# path = '/home/project_62/v_features'
# feats = load_feats_from(path, n_feat)
    
# generate random feats 
feats = [np.random.rand(T,n_feat) for i in range(T)]

inputs =  preprocess_embeddings(feats, n_feat, T)

model = AEwithAttention(n_feat, T, n_filt)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = True)

# training
for epoch in range(n_epochs):
    counter = 1
    for x in inputs:
        # Forward pass
        x = torch.tensor(x).float()
        dim = x.shape
        x_reconst = model(x).reshape(dim[0], dim[1])

        # Compute recons loss 
        loss_recons = criterion(x_reconst, x)
        
        #######
        # the following losses require paired video/caption data (x.v and x.t)
        # model_v and model_t are the corresponding models for video and captions respectively
        
        # Compute joint loss
        # loss_joint = criterion(model_v.encoder(x.v), model_t.encoder(x.t))
        
        # Compute cross loss
        # loss_cross1 = criterion(model_t.decoder(model_v.encoder(x.v)), x.t)
        # loss_cross2 = criterion(model_v.decoder(model_t.encoder(x.t)), x.v)
        # loss_cross = loss_cross1 + loss_cross2
        
        # Compute cycle loss
        # loss_cycle1 = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(x.t)))), x.t)
        # loss_cycle2 = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(x.v)))), x.v)
        # loss_cycle = loss_cycle1 + loss_cycle2
        
	# set hyperparams 
	# a1 = 0.1
	# a2 = 0.1
	# a3 = 0.1

        # Compute total loss
        # loss = loss_recons + a1 * loss_joint + a2 * loss_cross + a3 * loss_cycle
        #######
        
        # Backprop and optimize
        loss = loss_recons
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch[{}/{}], Step[{}/{}] Reconst Loss: {}\n'.format(epoch + 1, n_epochs, counter, len(inputs), loss.item()))

        counter = counter + 1

#torch.save(model.state_dict(), 'out/model.sd')
