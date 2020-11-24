import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM
import time


def _craft_advs(model, images, labels, epsilon, attack):    
    #preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    model.eval()
    fmodel = PyTorchModel(model, bounds=(0, 1))
    # apply the attack
    if attack == 'FGSM':
        attack = FGSM()
    if attack == 'PGD':
        attack = LinfPGD()
    
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilon)
    model.train()
    return clipped_advs



def _evaluate_model(model, data_loader, device, criterion):
    correct = 0
    total = 0
    acc_loss = 0.0
    avg_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            #print(i)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion != None:
                loss = criterion(outputs, labels)
                acc_loss += loss.item() 
                avg_loss = acc_loss / (i+1)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, avg_loss

def evaluate_clean_accuracy(model, data_loader, device, criterion=None):
    return _evaluate_model(model, data_loader, device, criterion)

def evaluate_rob_accuracy(model, data_loader, device, epsilon, attack):
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        adv_images = _craft_advs(model, images, labels, epsilon, attack)
        adv_images, labels = adv_images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(adv_images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def _evaluate_sparsity(model):
    if model.conv_masks == None and model.fully_connected_masks == None:
        model.identify_layers()
    params = model.state_dict()
    sparsities = {}
    total_no_of_non_zero, total_no_of_weights = 0, 0
    for mask in model.conv_masks+model.fully_connected_masks:
        layer_mask = params[mask]
        no_of_non_zero_weights = torch.nonzero(layer_mask.flatten(), as_tuple=False).nelement()
        no_of_weights = layer_mask.nelement()
        sparsity = 1 - no_of_non_zero_weights / no_of_weights
        sparsities[mask] = sparsity
        
        total_no_of_non_zero +=  no_of_non_zero_weights
        total_no_of_weights += no_of_weights
        
    return (sparsities, 1 - total_no_of_non_zero / total_no_of_weights)

def _identify_layers(model):
    conv_weights, conv_masks, fully_connected_weights, fully_connected_masks = [], [], [], []
    for key in model.state_dict().keys():
        if key[:1] == 'c' and 'weights' in key:
            conv_weights.append(key)
        if key[:1] == 'c' and 'mask' in key:
            conv_masks.append(key)
        if key[:2] == 'fc' and 'weights' in key:
            fully_connected_weights.append(key)
        if key[:2] == 'fc' and 'mask' in key:
            fully_connected_masks.append(key)
    return conv_weights, conv_masks, fully_connected_weights, fully_connected_masks

def safe_model(PATH, model, optimizer, description='N/A', loss='N/A',epoch='N/A'):
    torch.save({
        'description': description,
        'epoch': epoch,
        'train_stats': model.train_stats,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, PATH)
    return PATH

def load_model(model, PATH, optim='ADAM'):
    if optim=='ADAM':
        optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train_stats = checkpoint['train_stats']
    model.train()
    return model