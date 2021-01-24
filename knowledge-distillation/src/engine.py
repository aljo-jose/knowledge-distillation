import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer):
    model.train()
    final_loss = 0
    #for bi, (data,targets) in enumerate(islice(data_loader,5)):
    for bi, (data,targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        #print('train loss :', float(loss.item()))
        final_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    return final_loss/max(bi,1)
        
def eval_fn(data_loader, model):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        #for bi, (data,targets) in enumerate(islice(data_loader,5)):
        for bi, (data,targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            #print('test loss :', float(loss.item()))
            final_loss += float(loss.item())
    return final_loss/max(bi,1)

def check_accuracy(model, dataloader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for bi, (data,targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            outputs = model(data)
            predictions.extend(outputs.argmax(dim=-1).numpy().tolist())
            labels.extend(targets.numpy().tolist())
            
            correct_predictions_count = (np.asarray(predictions) == np.asarray(labels)).sum()
            incorrect_predictions_count = len(labels)-correct_predictions_count
            accuracy = correct_predictions_count / len(labels)
    return incorrect_predictions_count,accuracy