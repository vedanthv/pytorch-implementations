import torch

x = torch.arange(9)

x_3x3 = x.view(3,3) # acts on contiguous tensors
x_3x3 = x.reshape(3,3) # acts on non contiguous tensors

y = x_3x3.t() # [0,3,6,1,4,7,2,5,8]
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))

print(torch.cat((x1,x2),dim = 0).shape) # add across rows
print(torch.cat((x1,x2),dim = 1).shape) # add across columns

# Unrolling
z = x1.view(-1)
print(z.shape)

batch = 64
x=torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

z = x.premute(0,-2,1)

x = torch.arange(10) # [10]
print(x.unsqueeze(0).shape) # 1 x 10 
print(x.unsqueeze(1).shape) # 10 x 1
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1 x 1 x 10

z = x.unsqueeze(1)
print(x.shape)






