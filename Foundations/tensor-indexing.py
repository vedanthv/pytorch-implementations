import torch

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

print(x[0].shape)
print(x[:,0].shape)
print(x[2,0:10]) # 0:10 ---> [0,1,2,...9]

x[0,0] = 100

# Fancy INdexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x= torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])

# More advanced indexing
x = torch.arange(10)
print(x[x < 2] | x[x > 8])
print(x[x.remainder(2) == 0])

# Useful Operations
print(x.where(x>5,x,x+2))
print(torch.tensor([1,2,3,4,4].unique()))
print(x.ndimension())
print(x.numel) # Number of elements in x
