```python
import torch
import torch.nn as nn
```
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(666)
if device =='cuda':
  torch.cuda.manual_seed_all(666)
```
```python
X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)

model = nn.Sequential(
        nn.Linear(2,10,bias = True),
        nn.Sigmoid(),
        nn.Linear(10,10,bias = True),
        nn.Sigmoid(),
        nn.Linear(10,10,bias=True),
        nn.Sigmoid(),
        nn.Linear(10,1,bias=True),
        nn.Sigmoid()).to(device)

```

```python
criteridon = nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
```

```python
for epoch in range(100001):
  optimizer.zero_grad()
  hypothesis = model(X)
 
  cost = criteridon(hypothesis,Y)
  cost.backward()
  optimizer.step()
  
  if epoch % 100 == 0:
    print(epoch,cost.item())
```

```python
with torch.no_grad():
  hypothesis = model(X)
  predicted = (hypothesis > 0.5).float()
  accuracy = (predicted == Y).float().mean()
```
