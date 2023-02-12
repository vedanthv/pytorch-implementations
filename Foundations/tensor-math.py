# Tensor math and comparision
import torch

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z = x+y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x,y)

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x.pow(2)
print(z)
z = x ** 2

# Simple Comparision
z = x > 0
print(z)
z = x < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = x1.mm(x2)

# Matrix Exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

# elementwise multiplication
z = x+y
print(z)

# dot product
z = torch.dot(x,y) # sum of 21,16,9

# Batch Matrix Multiplication
batch = 32  
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,m,n))
tensor2 = torch.rand((batch,m,p))
out_bm = torch.bmm(tensor1,tensor2)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 - torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x,dim = 0)
values,indices = torch.max(x,dim = 0)
values,indices = torch.min(x,dim = 0)
z = torch.argmax(x,dim = 0)
mean_x = torch.mean(x.float(),dim = 0)
z = torch.eq(x,y)
sorted_y,indices = torch.sort(y,dim = 0,descending = False)

z = torch.clamp(x,min = 0) # All elements of x < 0 are set to 0

x = torch.tensor([1,0,1,1,1],dtype = torch.bool)
z = torch.any(x)
z = torch.all(x)

