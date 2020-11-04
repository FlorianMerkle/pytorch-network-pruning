import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, train_data, test_data, epochs, device):
    for epoch in range(epochs):  # loop over the dataset multiple times
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            #if i == 0:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            preds = torch.argmax(F.softmax(outputs, dim=1),dim=1)
            accuracy = int(sum(([pred == labels[i] for i, pred in enumerate(preds)])))/len(preds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

                # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss))
            print('[%d, %5d] test_accuracy: %.2f' %(epoch + 1, i + 1, accuracy))
            running_loss = 0.0
        accuracy = evaluate_model(model, test_data, device)
        print('val_accuracy: ', accuracy)
        
    print('Finished Training')
    return True
def evaluate_model(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print(i)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy