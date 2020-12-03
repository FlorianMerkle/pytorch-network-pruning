from . helpers import _craft_advs, _evaluate_model
import time
import torch
import torch.nn as nn
import torch.optim as optim
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack

def _fit_adv(model, train_data, test_data, epochs, device, attack, epsilon, patience=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    total_time = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
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
        total_time +=t1-t0
        accuracy, loss = _evaluate_model(model, test_data, device, criterion)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
    print('Finished Training')


    return model.train_stats

def _fit(model, train_loader, val_loader, epochs, device, patience=None, evaluate_robustness=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    total_time = 0
    epochs_trained = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time.time()
        acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0
        
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            batchsize = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / batchsize
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
        total_time += t1 - t0
        accuracy, loss = _evaluate_model(model, test_data, device, criterion)
        #print('duration:', t1-t0,'- train loss: ',avg_epoch_loss,' - train accuracy: ',avg_epoch_accuracy,' - validation accuracy: ', accuracy,' - validation loss: ', loss)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
        data = {
            'epoch': epoch+1,
            'train_loss':avg_epoch_loss, 
            'train_accuracy':avg_epoch_accuracy,
            'validation_loss':loss,
            'validation_accuracy':accuracy,
            'duration':total_time,
            'criterion':criterion,
            'optimizer':optimizer,
            'method': 'standard',
            'batchsize': len(next(iter(train_loader))[1])
        }
        
        
        if epoch%3==0 and evaluate_robustness == True:
            (l_0_robustness, l_0_loss), (l_2_robustness, l_2_loss), (l_inf_robustness, l_inf_loss) = _evaluate_robustness(model, val_loader, device)
            date['l_0_robustness'] = l_0_robustness
            date['l_2_robustness'] = l_2_robustness
            date['l_inf_robustness'] = l_inf_robustness
        
        model.train_stats = model.train_stats.append(data, ignore_index=True)
        
        if patience != None and patience < epoch and stop_early(val_loss_hist, patience) == True:
            epochs_trained = i + 1
            print('stopped early after', patience, 'epochs without decrease of validation loss')
            break
    print('Finished Training')
    
    return model.train_stats

def _fit_free(model, train_loader, val_loader , epochs, device, number_of_replays=3, eps = 16/255, patience=None, evaluate_robustness=False):
    #mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #mean = torch.tensor(mean).view(3,1,1).expand(3,32,32).to(device)
    #std = torch.tensor(std).view(3,1,1).expand(3,32,32).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    epochs_trained = 0
    total_time = 0
    pert_storage = torch.zeros((512,3,32,32)).to(device)
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
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
                #adv_input.sub_(mean).div_(std)
                
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
        total_time += t1 - t0
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
        if patience != None and patience < epoch and stop_early(val_loss_hist, patience) == True:
            print('stopped early after', patience, 'epochs without decrease of validation loss')
            epochs_trained = i + 1
            break
    print('Finished Training')
    return {
        'epochs_trained':epochs_trained,
        'avg_time_per_epoch': total_time/epochs,
        'criterion': criterion,
        'optimizer': optimizer,
        'hist': {
            'train_loss': train_loss_hist,
            'train_accuracy': train_acc_hist,
            'validation_loss': val_loss_hist,
            'validation_accuracy': val_acc_hist
        },
        'val_accuracy': accuracy
    }

def _fit_fast(model, train_loader, val_loader , epochs, device, eps = 8/255, patience=None, evaluate_robustness=False):
    #mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #mean = torch.tensor(mean).view(3,1,1).expand(3,32,32).to(device)
    #std = torch.tensor(std).view(3,1,1).expand(3,32,32).to(device)
    if model.optim == None:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = model.optim
    criterion = nn.CrossEntropyLoss().to(device)
    
    epochs_trained = 0
    total_time = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    for epoch in range(epochs):
        t0 = time.time()
        running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i, data in enumerate(train_loader):
            if i==i:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                pert = torch.zeros_like(inputs).uniform_(-eps, eps)
                pert.requires_grad = True
                adv_inputs = inputs + pert
                adv_inputs.clamp_(0, 1.0)
                #adv_inputs.sub_(mean).div_(std)
                #clip 0,1

                # first backwards pass to perform fgsm
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                alpha = 1.25*eps
                pert = pert + (alpha * torch.sign(pert.grad))
                pert.clamp_(-eps, eps)
                adv_inputs = inputs + pert
                adv_inputs.clamp_(0, 1.0)

                # second backwards pass to update weights on adv.
                optimizer.zero_grad()
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                clean_outputs = model(inputs)
                clean_train_accuracy = get_accuracy(labels, clean_outputs)
                adv_train_accuracy = get_accuracy(labels, outputs)
                acc_epoch_loss += loss.item() 
                avg_epoch_loss = acc_epoch_loss / (i+1)
                acc_epoch_accuracy += adv_train_accuracy
                avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
                if i%5 == 0:
                    print('[%d, %5d] loss: %.5f, adv_train_accuracy: %.2f, clean_train_accuracy : %.2f' %(epoch + 1, i + 1, loss.item(), adv_train_accuracy, clean_train_accuracy))
        f_adv, f_success = FGSM(model, val_loader, torch.nn.CrossEntropyLoss(), 8/255, device)
        print('fgsm robustness:',1-f_success)
        p_adv, p_success = PGD(model, val_loader, torch.nn.CrossEntropyLoss(), device)
        print('pgd robustness:', 1-p_success)

        t1 = time.time()
        total_time += t1 - t0
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)

        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.5f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
        data = {
            'epoch': epoch+1,
            'train_loss':avg_epoch_loss, 
            'train_accuracy':avg_epoch_accuracy,
            'validation_loss':loss,
            'validation_accuracy':accuracy,
            'duration':total_time,
            'criterion':criterion,
            'optimizer':optimizer,
            'method': 'standard',
            'batchsize': len(next(iter(train_loader))[1]),
            'fgsm': 1-f_success,
            'pgd_robustness': 1-p_success,
        }
        del f_adv
        del f_success
        del p_adv
        del p_success
        model.train_stats = model.train_stats.append(data, ignore_index=True)
        
        if patience != None and patience < epoch and stop_early(val_loss_hist, patience) == True:
            print('stopped early after', patience, 'epochs without decrease of validation loss')
            epochs_trained = i + 1
            break
    model.optim = optimizer
    print('Finished Training')
    return model.train_stats

        
def _fit_fast_with_double_update(model, train_loader, val_loader , epochs, device, eps = 8/255, patience=None, evaluate_robustness=False):
    #mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #mean = torch.tensor(mean).view(3,1,1).expand(3,32,32).to(device)
    #std = torch.tensor(std).view(3,1,1).expand(3,32,32).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    
    epochs_trained = 0
    total_time = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    for epoch in range(epochs):
        t0 = time.time()
        running_loss, acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pert = torch.rand_like(inputs, requires_grad=True)
            adv_inputs = inputs + pert
            adv_inputs.clamp_(0, 1.0)
            #adv_inputs.sub_(mean).div_(std)
            #clip 0,1
            
            # first backwards pass to perform fgsm
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            alpha = 1.25 * eps
            pert = pert + (alpha * torch.sign(pert.grad))
            pert.clamp_(-eps, eps)
            adv_inputs = inputs + pert
            adv_inputs.clamp_(0, 1.0)
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
        total_time += t1 - t0
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)

        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.5f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
        data = {
            'epoch': epoch+1,
            'train_loss':avg_epoch_loss, 
            'train_accuracy':avg_epoch_accuracy,
            'validation_loss':loss,
            'validation_accuracy':accuracy,
            'duration':total_time,
            'criterion':criterion,
            'optimizer':optimizer,
            'method': 'standard',
            'batchsize': len(next(iter(train_loader))[1])
        }
        
        
        if epoch%3==0 and evaluate_robustness == True:
            (l_0_robustness, l_0_loss), (l_2_robustness, l_2_loss), (l_inf_robustness, l_inf_loss) = _evaluate_robustness(model, val_loader, device)
            date['l_0_robustness'] = l_0_robustness
            date['l_2_robustness'] = l_2_robustness
            date['l_inf_robustness'] = l_inf_robustness
        model.train_stats = model.train_stats.append(data, ignore_index=True)
        
        if patience != None and patience < epoch and stop_early(val_loss_hist, patience) == True:
            epochs_trained = i + 1
            print('stopped early after', patience, 'epochs without decrease of validation loss')
            break
    print('Finished Training')
    return model.train_stats


### Helpers
def _evaluate_robustness(model, test_data, device):
    def bb_attack(model, images, labels, eps=8/255):
        model.eval()
        fmodel = PyTorchModel(model, bounds=(0, 1))
        attack = L0BrendelBethgeAttack()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=eps)
        model.train()
        return (1 - torch.sum(success)/len(success)) / 100

    def cw_attack(model, images, labels, eps=2):
        model.eval()
        fmodel = PyTorchModel(model, bounds=(0, 1))
        attack = L2CarliniWagnerAttack()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=eps)
        model.train()

        return (1 - torch.sum(success)/len(success))

    def pgd_attack(model, images, labels, eps=8/255):
        model.eval()
        fmodel = PyTorchModel(model, bounds=(0, 1))
        attack = LinfPGD()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=eps)
        model.train()
        return (1 - torch.sum(success)/len(success))
    
    images, labels = next(iter(test_data))
    images, labels = images.to(device), labels.to(device)
    l_0_robustness = 0
    l_2_robustness = cw_attack(model, images, labels)
    l_inf_robustness = pgd_attack(model, images, labels)
    return (l_0_robustness, 'n/a'),(l_2_robustness, 'n/a'),(l_inf_robustness, 'n/a')

def get_accuracy(labels, outputs):
    _, predicted = torch.max(outputs.data, 1)

    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def fgsm(gradients, step_size=.05):
    return step_size*torch.sign(gradients)

def stop_early(val_loss_hist, patience):
    return len(list(filter(lambda x: val_loss_hist[-patience-1] > x, val_loss_hist[-(patience):]))) == 0 # Check if any value in the last x-1 epochs is higher then the value of the epoch t-x 


def PGD(model, data_loader, criterion, device, max_stepsize=1.25*8/255, eps=8/255, steps=7):
    model.eval()
    advs = []
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        if i < 8:
            inputs, labels = data
            inputs, labels =inputs.to(device), labels.to(device)

            adv_examples = inputs
            adv_examples.requires_grad = True
            adv_examples.retain_grad()
            for step in range(steps):
                #print(torch.max(adv_examples[0]-inputs[0][0]))
                adv_examples, pert = FGSM_step(model, adv_examples, labels, criterion, max_stepsize, device)
                pert = adv_examples - inputs
                pert.clamp_(-eps, eps)
                adv_examples = inputs + pert
                adv_examples.clamp_(0,1)
            advs.append(adv_examples)
            preds = model(adv_examples)
            #pred_labels = 
            _, predicted = torch.max(preds.data, 1)
            total += len(predicted)
            #correct += (pred_labels == labels).sum().item()
            correct += (predicted != labels).sum().item()
    return advs, correct/total
        

def FGSM_step(model, inputs, labels, criterion, eps, device):

    inputs.retain_grad()
    perturbation = torch.zeros_like(inputs).to(device)
    preds = model(inputs)
    loss = criterion(preds, labels)
    loss.backward(retain_graph=True)
    perturbation = torch.sign(inputs.grad).clamp_(-eps, eps)
    adv_examples = inputs + perturbation
    adv_examples.clamp_(0,1)
    return adv_examples, perturbation
    

def FGSM(model, data_loader, criterion, eps, device):
    model.eval()
    #mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #mean = torch.tensor(mean).view(3,1,1).expand(3,32,32).to(device)
    #std = torch.tensor(std).view(3,1,1).expand(3,32,32).to(device)
    advs = []
    correct = 0
    total = 0
    for i,data in enumerate(data_loader):
        if i < 8:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            adv_examples, perturbation = FGSM_step(model, inputs, labels, criterion, eps, device)

            advs.append(adv_examples)
            preds = model(adv_examples)
            #pred_labels = 
            _, predicted = torch.max(preds.data, 1)
            total += len(predicted)
            #correct += (pred_labels == labels).sum().item()
            correct += (predicted != labels).sum().item()

    
    return advs, correct/total