import torch

# I have 
# export PYTHONPATH=~/mf-p2eft/src:$PYTHONPATH
# export PYTHONPATH=~/mf-p2eft:$PYTHONPATH
from models.minmaxplus.abx import TopKp
from models.pam.pam_ops import Linear

# Test
topg = TopK(10, 200, k=3)
pam_lin = Linear(10, 200)

x = torch.randn(128, 10)
print(topg(x).shape)
print(pam_lin(x).shape)
