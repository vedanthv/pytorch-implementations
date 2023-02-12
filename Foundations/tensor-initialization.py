import torch

device = "cuda" if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float32,device = device,requires_grad = True)
print(my_tensor)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initializations
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5)
x = torch.arange(start = 0,end = 5,step=1)
x = torch.empty(size = (1,5)).normal_(mean = 0,std = 1)
x = torch.diag(torch.ones(3))

# Convert between types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short()) # int 16 type
print(tensor.long()) # int 64 bit
print(tensor.float()) # float 32

# Array to Tensor
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()




