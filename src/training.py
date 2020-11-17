from . helpers import _craft_advs, _evaluate_model
import time
import torch
import torch.nn as nn
import torch.optim as optim

def _fit_adv(model, train_data, test_data, epochs, device, attack, epsilon):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time.time()
        running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(train_data, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            #if i == 0:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            adv_inputs = _craft_advs(model, inputs, labels, epsilon, attack)
            combined_inputs = torch.cat((inputs, adv_inputs))
            combined_labels = torch.cat((labels, labels))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(combined_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = combined_labels.size(0)
            correct = (predicted == combined_labels).sum().item()
            accuracy = 100 * correct / total
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

                # print statistics
            running_loss += loss.item()
            acc_epoch_loss += running_loss 
            avg_epoch_loss = acc_epoch_loss / (i+1)
            acc_epoch_accuracy += accuracy
            avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
            if i%10 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, running_loss, accuracy))
            running_loss = 0.0
        t1 = time.time()
        accuracy, loss = _evaluate_model(model, test_data, device, criterion)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        
    print('Finished Training')
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': 'Not implemented',
        'val_accuracy': accuracy
    }

def _fit(model, train_data, test_data, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time.time()
        acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0

        
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            #if i == 0:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

                # print statistics
            acc_epoch_loss += loss.item() 
            avg_epoch_loss = acc_epoch_loss / (i+1)
            acc_epoch_accuracy += accuracy
            avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
            if i%10 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, loss.item(), accuracy))
        t1 = time.time()
        accuracy, loss = _evaluate_model(model, test_data, device, criterion)
        #print('duration:', t1-t0,'- train loss: ',avg_epoch_loss,' - train accuracy: ',avg_epoch_accuracy,' - validation accuracy: ', accuracy,' - validation loss: ', loss)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        
    print('Finished Training')
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': 'Not implemented',
        'val_accuracy': accuracy
    }

def _fit_free(model, train_loader, val_loader , epochs, device, number_of_replays=3, eps = 16/255):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = torch.tensor(mean).view(3,1,1).expand(3,32,32)
    std = torch.tensor(std).view(3,1,1).expand(3,32,32)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    
    pert_storage = torch.zeros((512,3,32,32))
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time.time()
        acc_accuracy, acc_loss, running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        #pert_storage = torch.zeros([512, 3, 32,32])
        for i, data in enumerate(train_loader, 0):
            mini_batch_loss = 0.0
            mini_batch_acc = 0.0
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            no_of_samples_in_batch = inputs.shape[0]
            

            # Mini Batch Replays
            for j in range(number_of_replays):
                noise_batch = pert_storage[:no_of_samples_in_batch].detach().clone().requires_grad_(True)
                adv_input = inputs+noise_batch[:no_of_samples_in_batch]
                
                adv_input.clamp_(0, 1.0)
                adv_input.sub_(mean).div_(std)
                
                #print(adv_input[0])

                # forward + backward + optimize
                outputs = model(adv_input)
                loss = criterion(outputs, labels)
                
                # zero the gradients
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                
                #craft adv pert
                pert = fgsm(noise_batch.grad)
                
                pert_storage[0:no_of_samples_in_batch] += pert
                pert_storage.clamp_(-eps, eps)
                #update weights
                optimizer.step()

                    # print statistics
                accuracy = get_accuracy(labels, outputs)
                mini_batch_loss += loss.item()
                mini_batch_acc += accuracy
                acc_loss += mini_batch_loss
                acc_accuracy += mini_batch_acc
                
            avg_mini_batch_accuracy = mini_batch_acc / number_of_replays    
            avg_mini_batch_loss = mini_batch_loss / number_of_replays
            if i%1 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, avg_mini_batch_loss, avg_mini_batch_accuracy))
            acc_epoch_accuracy += mini_batch_acc
            acc_epoch_loss += mini_batch_loss
        avg_epoch_accuracy = acc_epoch_accuracy / (i+1)*number_of_replays    
        avg_epoch_loss = acc_epoch_loss / (i+1)*number_of_replays
        t1 = time.time()
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)
        print('duration:', t1-t0,'- train loss: ',avg_epoch_loss,' - train accuracy: ',avg_epoch_accuracy,' - validation accuracy: ', accuracy,' - validation loss: ', loss)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        
    print('Finished Training')
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': 'Not implemented',
        'val_accuracy': accuracy
    }

def _fit_fast(model, train_loader, val_loader , epochs, device, eps = 8/255):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = torch.tensor(mean).view(3,1,1).expand(3,32,32)
    std = torch.tensor(std).view(3,1,1).expand(3,32,32)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        t0 = time.time()
        running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pert = torch.rand_like(inputs, requires_grad=True)
            adv_inputs = inputs + pert
            adv_inputs.clamp_(0, 1.0)
            adv_inputs.sub_(mean).div_(std)
            #clip 0,1
            
            # first backwards pass to perform fgsm
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            pert = pert + (eps * pert.grad)
            pert.clamp_(-eps, eps)
            adv_inputs = inputs + pert
            
            # second backwards pass to update weights on adv.
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            accuracy = get_accuracy(labels, outputs)
            acc_epoch_loss += loss.item() 
            avg_epoch_loss = acc_epoch_loss / (i+1)
            acc_epoch_accuracy += accuracy
            avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
            if i%10 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, loss.item(), accuracy))

        t1 = time.time()
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)

        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.5f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
    print('Finished Training')
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': 'Not implemented',
        'val_accuracy': accuracy
    }

        
def _fit_fast_with_double_update(model, train_loader, val_loader , epochs, device, eps = 8/255):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = torch.tensor(mean).view(3,1,1).expand(3,32,32)
    std = torch.tensor(std).view(3,1,1).expand(3,32,32)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        t0 = time.time()
        running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pert = torch.rand_like(inputs, requires_grad=True)
            adv_inputs = inputs + pert
            adv_inputs.clamp_(0, 1.0)
            adv_inputs.sub_(mean).div_(std)
            #clip 0,1
            
            # first backwards pass to perform fgsm
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            pert = pert + (eps * pert.grad)
            pert.clamp_(-eps, eps)
            adv_inputs = inputs + pert
            optimizer.step()
            
            # second backwards pass to update weights on adv.
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            accuracy = get_accuracy(labels, outputs)
            acc_epoch_loss += loss.item() 
            avg_epoch_loss = acc_epoch_loss / (i+1)
            acc_epoch_accuracy += accuracy
            avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
            if i%10 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, loss.item(), accuracy))

        t1 = time.time()
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)

        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.5f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
    print('Finished Training')
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': 'Not implemented',
        'val_accuracy': accuracy
    }


### Helpers

def get_accuracy(labels, outputs):
    _, predicted = torch.max(outputs.data, 1)

    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def fgsm(gradients, step_size=.05):
    return step_size*torch.sign(gradients)