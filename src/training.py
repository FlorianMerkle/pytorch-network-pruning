from . helpers import _craft_advs, _evaluate_model
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack
import copy

def clamp(X, lower_limit, upper_limit):
    #print(upper_limit.shape)
    return torch.max(torch.min(X, upper_limit), lower_limit)


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        if i < 2:
            X, y = X.cuda(), y.cuda()
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    model.train()
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

std = torch.tensor((1,1,1)).view(3,1,1).cuda()
std = torch.tensor((1,1,1)).view(3,1,1).cuda()
#epsilon = eps/255 / std
#alpha = alpha/255 / std
pgd_alpha = (2 / 255.) / std
lr_min = 0.
lr_max = 0.2
momentum = .9
weight_decay = 5e-4
lower_limit = 0
upper_limit = 1


def _fit_fast_new(model, train_loader, test_loader, device, pruning_ratio=0, pruning_steps=1, epochs = 20, epsilon = 8, alpha = 10, pgd_alpha = 2, lr_min = 0.,lr_max = 0.2, momentum = .9, weight_decay = 5e-4):
    std = torch.tensor((1,1,1)).view(3,1,1).cuda()
    pgd_alpha = (pgd_alpha / 255.) / std
    lower_limit = torch.tensor((0,0,0)).view(3,1,1).cuda()
    upper_limit = torch.tensor((1,1,1)).view(3,1,1).cuda()
    epsilon = (epsilon / 255.) / std
    alpha = (alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std


    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    #amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    #if args.opt_level == 'O2':
    #    amp_args['master_weights'] = args.master_weights
    #model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()


    lr_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    
    pruning_intervals = round(epochs/(pruning_steps + 1))
    print(pruning_intervals)
    pruning_schedule = [ epoch % pruning_intervals == 0 and epoch / pruning_intervals != 0 for epoch in list(range(epochs))]
    print(pruning_schedule)
    pruning_step = 1
    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    #logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        if pruning_schedule[epoch] == True:
            pruning_step_ratio = pruning_ratio/pruning_steps*pruning_step
            print(pruning_step_ratio)
            model.prune_magnitude_global_unstruct(pruning_step_ratio, device)
            pruning_step+=1
            
        print('start epoch:', epoch)
        
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            delta = torch.zeros_like(X).cuda()
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            #with amp.scale_loss(loss, opt) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            #print(type(lower_limit))
            #print(type(X))
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            #with amp.scale_loss(loss, opt) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        
            # Check current PGD robustness of model using random minibatch
#        X, y = first_batch
#        pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 7, 1, opt)
#        with torch.no_grad():
#            output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
        robust_acc = evaluate_pgd(test_loader, model, 7, 1)[1]
        print('robustness: ',robust_acc, )
        if robust_acc - prev_robust_acc < -0.2:
            break
        if robust_acc > prev_robust_acc:
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        #logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f', epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        print(epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    
    #best_state_dict = model.state_dict()
    #torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
    #logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = CifarResNet().cuda()
    model.load_state_dict(best_state_dict)
    model.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 30, 3)
    test_loss, test_acc = evaluate_standard(test_loader, model)

    #logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    #logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    
    
    return


def _fit_fast_locuslab(model, train_loader, val_loader , epochs, device, alpha=10, eps = 8, number_of_replays=7, patience=None, evaluate_robustness=False):
    hist = []
    std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
    epsilon = eps/255. / std
    alpha = alpha/255. / std
    pgd_alpha = (2 / 255.) / std
    lr_min = 0.
    lr_max = 1e-2
    momentum = .9
    weight_decay = 5e-4
    epochs = epochs
    lower_limit = 0.
    upper_limit = 1.
    model.train()
    
    
    opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)    
        
        
    criterion = nn.CrossEntropyLoss()
    lr_steps = epochs * len(train_loader)

    #scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps /(50-x) for x in range(50)], gamma=0.1)

        
    prev_robust_acc = 0. 
    state_dicts = []
    
    for epoch in range(epochs):
        print('start epoch:', epoch)
        start_epoch_time = time.time()       
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            if i == 0:
                #print('first batch',X.shape)
                first_batch = (X, y)
            delta = torch.zeros_like(X).cuda()
            #for j in range(len(epsilon)):
            #    if i == 0:
            #        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            #delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        X, y = first_batch
        #print('pre pgd',X.shape)
        pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 20, 1, opt)
        with torch.no_grad():
            model.eval()
            output = model(clamp(X + pgd_delta[:X.size(0)], torch.tensor((0,0,0)).view(3,1,1).cuda(), torch.tensor((1,1,1)).view(3,1,1).cuda()))
            model.train()
        #robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
        _, robust_acc = evaluate_pgd([(X,y)], model, 7,1)
        print(robust_acc, robust_acc - prev_robust_acc, robust_acc - prev_robust_acc < -0.2)
        hist.append(
            {
                'epoch':epoch+1,
                'clean accuracy':train_acc/train_n,
                'robust accuracy':robust_acc,
                'state dict': copy.deepcopy(model.state_dict())
            }
        )
        if robust_acc - prev_robust_acc < -0.2:
            break
        if robust_acc > prev_robust_acc:
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        state_dicts.append(copy.deepcopy(model.state_dict()))
        lr = scheduler.get_last_lr()[0]
        epoch_time = time.time()
        #print('epoch time:', start_epoch_time - epoch_time)
        print(epoch, lr, train_loss/train_n, train_acc/train_n)
    print('training finished')
    
    model.load_state_dict(best_state_dict)
    model.eval()

    #print(evaluate_pgd(val_loader, model, 30, 3)[1])
    #print(evaluate_standard(val_loader, model)[1])

    return hist

def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        if i == i:
            X, y = X.cuda(), y.cuda()
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    return pgd_loss/n, pgd_acc/n

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


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
        #optimizer = optim.Adam(model.parameters(), lr=5e-4)
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        
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
                #pert = torch.zeros_like(inputs).uniform_(-eps, eps)
                #pert.requires_grad = True
                #adv_inputs = inputs + pert
                #adv_inputs.clamp_(0, 1.0)

                
                
                pert = torch.zeros_like(inputs).uniform_(-eps, eps)
                
                pert.requires_grad = True
                pert.retain_grad()
                pert = (inputs+pert).clamp_(0, 1.0) - inputs
                adv_inputs = inputs + pert
                
                pert.retain_grad()
                #adv_inputs.sub_(mean).div_(std)
                #clip 0,1

                # first backwards pass to perform fgsm
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                pert.retain_grad()
                alpha = 1.25*eps
                pert = pert + (alpha * torch.sign(pert.grad))
                pert = (inputs + pert).clamp_(0,1)-inputs
                adv_inputs = inputs + pert

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