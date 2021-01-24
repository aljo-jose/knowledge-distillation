import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pathlib

import engine
import config
import models

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
train_dataset = datasets.MNIST('input/', train=True, download=True,
                   transform=transform)
test_dataset = datasets.MNIST('input/', train=False,
                   transform=transform)

print('train_dataset len' , len(train_dataset))
print('test_dataset len' , len(test_dataset))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

torch.manual_seed(42) # to reprduce the same results.
model = models.FCModel()
optimizer = optim.AdamW(model.parameters(), lr=0.005)
pathlib.Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)

n_epochs = 10
best_loss = np.inf 
for epoch in range(n_epochs):
    train_loss = engine.train_fn(train_data_loader, model, optimizer)
    test_loss = engine.eval_fn(test_data_loader,model)
    print(f'epoch {epoch+1}, train loss : {train_loss}, test loss : {test_loss}')
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), config.MODEL_PATH)
        print('model saved.')