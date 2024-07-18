import torch
from torch import nn


class L0raConfig:
    def __init__ 

# LoRA 0th reimplementation attempt
class L0ra(nn.Module):
    def __init__(
            self, base_model, 
        ):
        super(TopK, self).__init__()
        
        assert k or r, "either qty k or rate r must be defined"
        if k:
            assert k <= input_dim, "k shouldn't be greater than input dim"
            assert isinstance(k, int), "k must have an integer type"
        else:
            assert r > 0 and r <= 1, "extraction rate r should be in (0,1] interval"
            # https://ifunny.co/picture/i-know-i-rounded-them-up-Xp83yefS7
            assert round(input_dim*r) != 0, "extraction qty k = round(m*r) is 0 😢"
        assert f in ('-', 'abx', 'ba')
        
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.lbd = nn.Parameter(torch.Tensor(output_dim))
        self.T = T if not T_param else (
            nn.Parameter(torch.Tensor(1)) if T_param == 'common' else nn.Parameter(torch.Tensor(output_dim)) 
        )
        self.counts = nn.Parameter(torch.zeros(output_dim, input_dim))

        self.reset_parameters()
        
        self.k = round(input_dim*r) if r else k
        if is_adapt: self.k = adapt_k(self.k, C)
        
        # didnt test this yet
        self.activation = {
            'abx': self.abx,
            '-': self.sub,
            'ba': self.ba
        }[f]
        self.relu = torch.nn.ReLU()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)  # he-like, but it's wrong to my point of view
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.lbd, -bound, bound) 
        if not isinstance(self.T, float):  # initialization should be different 
            nn.init.uniform_(self.T, -bound, bound)            


    def count(self, indices):  # just a function to count everything
        exes = torch.arange(self.lbd.shape[0]).unsqueeze(1).to('cuda:0')
        for b in indices.to('cuda:0'):
            self.counts[exes, b] += 1


    def forward(self, input):
        """ It's pretty tricky here.
        Input ~ (*, N), W ~ (M,N)
        (*, N) -> (*, 1, N)
        (M,N) -> (1, M,N)
        topk( (*,1,N) + (1,M,N), k, dim=-1) ~ (*,M,k)
        sum( (*,M,k) , dim=-1) = (*,M)
        """
        # print(input.unsqueeze(-2).shape, self.weight.unsqueeze(0).shape)
        added = input.unsqueeze(-2) + self.weight.unsqueeze(0)
        # extr = added.topk(self.k, dim=2)[0].sum(dim=2)  # Sum may be replaced w mean        
        extr = added.topk(self.k, dim=-1)[0].sum(dim=-1)  # Sum may be replaced w mean        
        return self.activation(extr)


    # Activation functions 
    def sub(self, extr):
        return torch.relu(extr-self.T)*self.lbd + self.bias

    def abx(self, extr):
        acts  = extr > self.T
        output = acts*(torch.relu(extr)*self.lbd + self.bias)
        return torch.clamp(output, max=1e6) 

    def ba(self, extr):
        # acts  = extr > self.T
        # return self.bias + acts*extr    
        return self.bias + self.relu(extr - self.T)


    ### To be removed
    def _abx(self, input): 
        added = input.unsqueeze(1) + self.weight.unsqueeze(0)
        extr = added.topk(self.k, dim=2)[0].sum(dim=2)  # Sum may be replaced w mean        
        acts  = extr > self.T
        output = acts*(torch.relu(extr)*self.lbd + self.bias)
        return torch.clamp(output, max=1e6) 
                    
    def _sub(self, input):
        added = input.unsqueeze(1) + self.weight.unsqueeze(0)
        extr = added.topk(self.k, dim=2)[0].sum(dim=2)  # Sum may be replaced w mean
        return torch.relu(extr-self.T)*self.lbd + self.bias
    
    def _ba(self, input):
        added = input.unsqueeze(1) + self.weight.unsqueeze(0)
        choices = added.topk(self.k, dim=2)
        extr = choices[0].sum(dim=2)  # Sum may be replaced w mean
        self.count(choices[1])
        acts  = extr > self.T
        return self.bias + acts*extr    
    ###


    def train(self, mode=True):  # trick to avoid updating of counts matrix
        super(TopK, self).train(mode)
        self.counts.requires_grad = False
    
