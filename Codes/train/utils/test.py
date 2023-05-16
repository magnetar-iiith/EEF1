if(False):
    print('noting')

import torch
a = torch.rand(7,10,20)
b = torch.rand(7,5,10,20)

c = torch.cat((a[:,None,:,:],b[:,4:,:,:]),1)
print(c.shape)