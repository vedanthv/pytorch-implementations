import torch
from d2l import torch as d2l

# Defining Model
class LinearRegression(d2l.Module):
    def __init__(self,num_inputs,lr,sigma=0.01):
        super.__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0,sigma,(num_inputs,1),requires_grad = True)
        self.b = torch.zeros(1,requires_grad = True)

# Forward Prop Algorithm
@d2l.add_to_class(LinearRegression)
def forward(self,X):
    return torch.matmul(X,self.w)+self.b

# Loss Function
@d2l.add_to_class(LinearRegression)  
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean() # avg loss over all examples in mini batch

# Defining the Optimization Algorithm
class SGD(d2l.HyperParameters):
    def __init__(self,params,lr):
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            params -= self.lr * param.grid # used to iterate over parameter values
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Instance of SGD Class
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

# Training

@d2l.add_to_class(d2l.trainer)
def prepare_batch(self,batch):
    return batch

@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val,self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for barch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1

model = LinearRegression(2,lr = 0.03)
data = d2l.SyntheticRegression(w = torch.tensor([2,-3.4]),b = 4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model,data)

print(f'Error in estimating w:{data.w - model.w.reshape(data.w.shape)}')

print(f'error in estimating b is {data.b-model.b}')
