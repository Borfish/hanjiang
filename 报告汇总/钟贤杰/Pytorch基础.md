# 层和块


```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```




    tensor([[ 0.3412,  0.2021,  0.1343, -0.2887, -0.1175,  0.2535, -0.0106,  0.0869,
              0.1650, -0.1183],
            [ 0.2806,  0.1156,  0.0821, -0.1544, -0.0565,  0.2060, -0.0164, -0.0220,
              0.0776, -0.2116]], grad_fn=<AddmmBackward0>)



### 自定义块


```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

### 实例化多层感知机的层，然后再每次调用正向传播函数时调用这些层


```python
net = MLP()
net(X)
```




    tensor([[-0.1197,  0.0806,  0.0614, -0.0231, -0.2443,  0.0808,  0.0873, -0.1351,
             -0.0407, -0.1689],
            [-0.0432,  0.1223,  0.0185, -0.1017, -0.1646,  0.1151,  0.0194, -0.1507,
             -0.0787, -0.2064]], grad_fn=<AddmmBackward0>)



### 顺序块


```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block
        
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```




    tensor([[ 0.0113,  0.0712, -0.0587, -0.1847, -0.0469,  0.2205,  0.0749, -0.0337,
             -0.0079, -0.2208],
            [-0.1299,  0.0833, -0.1372, -0.2789, -0.0610,  0.2113,  0.2491, -0.0525,
              0.0472, -0.2906]], grad_fn=<AddmmBackward0>)



### 在正向传播函数中执行代码块


```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
net(X)
```




    tensor(-0.0465, grad_fn=<SumBackward0>)



### 混合搭配各种组合块的方法


```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))
    
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
net(X)
```




    tensor(-0.0465, grad_fn=<SumBackward0>)



# 参数管理


```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```




    tensor([[0.2120],
            [0.2623]], grad_fn=<AddmmBackward0>)



### 目标参数


```python
print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

    OrderedDict([('weight', tensor([[ 0.1905,  0.0346, -0.2435,  0.1668, -0.0394, -0.3113, -0.2796,  0.2530]])), ('bias', tensor([0.3354]))])
    <class 'torch.nn.parameter.Parameter'>
    Parameter containing:
    tensor([0.3354], requires_grad=True)
    tensor([0.3354])
    


```python
net[2].weight.grad == None
```




    True



### 一次性访问所以参数


```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

    ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
    ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
    


```python
net.state_dict()['2.bias'].data
```




    tensor([0.3354])



### 从嵌套块收集参数


```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```




    tensor([[0.4635],
            [0.4635]], grad_fn=<AddmmBackward0>)




```python
print(rgnet)
```

    Sequential(
      (0): Sequential(
        (block0): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block1): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block2): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block3): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
      )
      (1): Linear(in_features=4, out_features=1, bias=True)
    )
    

### 内置初始化


```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 初始化
        nn.init.ones_(m.bias)
        
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```




    (tensor([ 0.0041, -0.0104,  0.0031, -0.0071]), tensor(1.))




```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]        
        
```




    (tensor([1., 1., 1., 1.]), tensor(0.))




```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(xavier)
net[2].apply(init_42)
net[0].weight.data[0], net[2].weight.data  
```




    (tensor([-0.2937,  0.0213,  0.5936, -0.1529]),
     tensor([[42., 42., 42., 42., 42., 42., 42., 42.]]))



### 自定义初始化


```python
def my_init(m):
    if type(m) == nn.Linear:
        print(
        "init", 
        *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
        
net.apply(my_init)
net[0].weight[:2]
```

    init weight torch.Size([8, 4])
    init weight torch.Size([1, 8])
    




    tensor([[-0.0000,  5.9943, -6.8359, -0.0000],
            [-0.0000, -0.0000, -9.9551,  6.0796]], grad_fn=<SliceBackward0>)




```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```




    tensor([42.0000,  7.9943, -4.8359,  2.0000])



### 参数绑定


```python
# shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(),
                    shared, nn.ReLU(), nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
```

# 自定义层

### 不带任何参数的自定义层


```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
    
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```




    tensor([-2., -1.,  0.,  1.,  2.])



### 将层作为组件合并到构建更复杂的模型中


```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()
```




    tensor(9.3132e-10, grad_fn=<MeanBackward0>)



### 带参数的层


```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
linear = MyLinear(5, 3)
linear.weight
```




    Parameter containing:
    tensor([[ 0.2723,  0.1041, -0.7724],
            [-0.0246, -0.3009, -1.8623],
            [-0.2536, -1.1142, -0.0709],
            [ 1.4985,  0.2036,  0.4362],
            [ 0.7531,  0.2981, -0.7804]], requires_grad=True)



### 使用自定义层直接执行正向传播计算


```python
linear(torch.rand(2, 5))
```




    tensor([[1.3877, 0.0587, 0.0000],
            [2.6072, 0.5412, 0.0000]])



### 使用自定义层构建模型


```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```




    tensor([[3.3363],
            [8.2414]])




```python
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
x2
```




    tensor([0, 1, 2, 3])




```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```




    (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))




```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```




    {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}



### 加载和保存模型参数


```python
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
```

### 将模型参数储存为“mlp.params”文件
实例化原始模型备份，直接读取文件中储存的参数


```python
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```




    MLP(
      (hidden): Linear(in_features=20, out_features=256, bias=True)
      (out): Linear(in_features=256, out_features=10, bias=True)
    )




```python
Y_clone = clone(X)
Y_clone == Y
```




    tensor([[True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True]])



# GPU


```python
torch.device('cpu'), torch.cuda.device_count()
```




    (device(type='cpu'), 0)




```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```




    (device(type='cpu'), device(type='cpu'), [device(type='cpu')])




```python
x = torch.tensor([1, 2, 3])
x.device
```




    device(type='cpu')




```python
X = torch.ones(2, 3, device=try_gpu())
X
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])



### 神经网络与GPU


```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

net(X)
```




    tensor([[0.8205],
            [0.8205]], grad_fn=<AddmmBackward0>)



### 模型参数储存位置


```python
net[0].weight.data.device
```




    device(type='cpu')




```python

```
