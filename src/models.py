import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

from . custom_modules import MaskedLinearLayer, MaskedConvLayer, ResBlock
from . helpers import _identify_layers, _evaluate_sparsity
from . training import _fit, _fit_adv, _fit_free, _fit_fast, _fit_fast_with_double_update
from . pruning import _prune_random_local_unstruct, _prune_magnitude_global_unstruct, _prune_random_local_struct, _prune_random_global_struct, _prune_magnitude_local_struct, _prune_magnitude_global_struct, _prune_magnitude_local_unstruct

class CifarResNet(nn.Module):
    def __init__(self):
        super(CifarResNet, self).__init__()
        self.c1 = MaskedConvLayer((64, 3, 3, 3), padding=(1,1),bias=False, stride=1)
        self.r1 = ResBlock(64,64,padding=1)
        self.r2 = ResBlock(64,64,padding=1)
        self.r3 = ResBlock(64,128,stride=2,padding=1)
        self.r4 = ResBlock(128,128,padding=1)
        self.r5 = ResBlock(128,256,stride=2,padding=1)
        self.r6 = ResBlock(256,256,padding=1)
        self.r7 = ResBlock(256,512,stride=2,padding=1)
        self.r8 = ResBlock(512,512,padding=1)
        self.p2 = nn.AvgPool2d(4)
        self.d1 = MaskedLinearLayer((512,10), activation=None)
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = None, None, None, None
        self.identify_layers()
        self.train_stats = pd.DataFrame(columns=('epoch', 'train_loss', 'train_accuracy', 'validation_loss','l_inf_robustness','l_inf_loss' ,'l_2_robustness','l_2_loss','l_0_robustness','l_0_loss', 'validation_accuracy', 'duration', 'criterion', 'optimizer', 'method', 'learning_rate', 'batchsize'))
    
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.p2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.d1(x)
        return (x)
    
    def fit(self, train_data, val_data, epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit(self, train_data, val_data, epochs, device, patience=patience)
    
    def fit_adv(self, train_data, test_data, epochs, device, epsilon, attack='PGD', eps = 8/255, number_of_replays=7, patience=None):
        return _fit_adv(self, train_data, test_data, epochs, device, attack, epsilon, patience=patience)
    
    def fit_free(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_free(self, train_loader, val_loader , epochs, device, number_of_replays, eps, patience=patience)
    
    def fit_fast(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def identify_layers(self):
        print('identifying layers')
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = _identify_layers(self)
        return self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks
    
    def evaluate_sparsity(self):
        return _evaluate_sparsity(self)
    
    def prune_random_local_unstruct(self, ratio, device):
        _prune_random_local_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_magnitude_global_unstruct(self, ratio, device):
        _prune_magnitude_global_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_random_global_struct(self, ratio, device, prune_dense_layers=False):
        _prune_random_global_struct(self, ratio, device, prune_dense_layers=False)
        return False

    def prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel'):
        _prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_local_unstruct(self, ratio, device, scope='layer'):
        _prune_magnitude_local_unstruct(self, ratio, device, scope='layer')
        return self.evaluate_sparsity()


class ImageNetResNet(nn.Module):
    def __init__(self):
        super(ImageNetResNet, self).__init__()
        self.c1 = MaskedConvLayer((64, 3, 7, 7), padding=(3,3),bias=False, stride=2)
        self.p1 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1))
        self.r1 = ResBlock(64,64,padding=1)
        self.r2 = ResBlock(64,64,padding=1)
        self.r3 = ResBlock(64,128,stride=2,padding=1)
        self.r4 = ResBlock(128,128,padding=1)
        self.r5 = ResBlock(128,256,stride=2,padding=1)
        self.r6 = ResBlock(256,256,padding=1)
        self.r7 = ResBlock(256,512,stride=2,padding=1)
        self.r8 = ResBlock(512,512,padding=1)
        self.p2 = nn.AvgPool2d(7)
        self.d1 = MaskedLinearLayer((512,10), activation=None)
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = None, None, None, None
        self.identify_layers()
        self.train_stats = pd.DataFrame(columns=('epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'duration', 'criterion', 'optimizer', 'method', 'learning_rate', 'batchsize'))
    
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.p2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.d1(x)
        return (x)
    
    def fit(self, train_data, val_data, epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit(self, train_data, val_data, epochs, device, patience=patience)
    
    def fit_adv(self, train_data, test_data, epochs, device, epsilon, attack='PGD', eps = 8/255, number_of_replays=7, patience=None):
        return _fit_adv(self, train_data, test_data, epochs, device, attack, epsilon, patience=patience)
    
    def fit_free(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_free(self, train_loader, val_loader , epochs, device, number_of_replays, eps, patience=patience)
    
    def fit_fast(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def identify_layers(self):
        print('identifying layers')
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = _identify_layers(self)
        return self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks
    
    def evaluate_sparsity(self):
        return _evaluate_sparsity(self)
    
    def prune_random_local_unstruct(self, ratio, device):
        _prune_random_local_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_magnitude_global_unstruct(self, ratio, device):
        _prune_magnitude_global_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_random_global_struct(self, ratio, device, prune_dense_layers=False):
        _prune_random_global_struct(self, ratio, device, prune_dense_layers=False)
        return False

    def prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel'):
        _prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_local_unstruct(self, ratio, device, scope='layer'):
        _prune_magnitude_local_unstruct(self, ratio, device, scope='layer')
        return self.evaluate_sparsity()
        
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.c1 = MaskedConvLayer((6, 1, 5, 5), padding=0, activation='relu')
        self.c2 = MaskedConvLayer((16, 6, 5, 5), padding=0, activation='relu')
        self.p1 = nn.AvgPool2d(2)
        self.p2 = nn.AvgPool2d(2)
        self.fc1 = MaskedLinearLayer((256, 128))
        self.fc2 = MaskedLinearLayer((128, 84))
        self.fc3 = MaskedLinearLayer((84, 10), activation=None)
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = None, None, None, None
        self.identify_layers()
        self.train_stats = pd.DataFrame(columns=('epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'duration', 'criterion', 'optimizer', 'method', 'learning_rate', 'batchsize'))
        
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def fit(self, train_data, val_data, epochs, device):
        _fit(self, train_data, val_data, epochs, device)
        return True
    
    def fit_adv(self, train_data, test_data, epochs, device, epsilon, attack='PGD'):
        _fit_adv(self, train_data, test_data, epochs, device, attack, epsilon)
        return True
    
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.c1 = MaskedConvLayer((16, 3, 3, 3), padding=0, activation='relu')
        self.c2 = MaskedConvLayer((16, 16, 3, 3), padding=0, stride=2, activation='relu')
        self.c3 = MaskedConvLayer((32, 16, 3, 3), padding=0, activation='relu')
        self.c4 = MaskedConvLayer((32, 32, 3, 3), padding=0, stride=2, activation='relu')
        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(16)
        self.b3 = nn.BatchNorm2d(32)
        self.b4 = nn.BatchNorm2d(32)
        self.fc2 = MaskedLinearLayer((800, 128))
        self.fc3 = MaskedLinearLayer((128, 10), activation=None)
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = None, None, None, None
        self.identify_layers()
        self.train_stats = pd.DataFrame(columns=('epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'duration', 'criterion', 'optimizer', 'method', 'learning_rate', 'batchsize'))
        
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def fit(self, train_data, val_data, epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit(self, train_data, val_data, epochs, device, patience=patience)
    
    def fit_adv(self, train_data, test_data, epochs, device, epsilon, attack='PGD', eps = 8/255, number_of_replays=7, patience=None):
        return _fit_adv(self, train_data, test_data, epochs, device, attack, epsilon, patience=patience)
    
    def fit_free(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_free(self, train_loader, val_loader , epochs, device, number_of_replays, eps, patience=patience)
    
    def fit_fast(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps = 8/255, number_of_replays=7, patience=None):
        return _fit_fast_with_double_update(self, train_loader, val_loader , epochs, device, eps, patience=patience)
    
    def identify_layers(self):
        print('identifying layers')
        self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks = _identify_layers(self)
        return self.conv_weights, self.conv_masks, self.fully_connected_weights, self.fully_connected_masks
    
    def evaluate_sparsity(self):
        return _evaluate_sparsity(self)
    
    def prune_random_local_unstruct(self, ratio, device):
        _prune_random_local_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_magnitude_global_unstruct(self, ratio, device):
        _prune_magnitude_global_unstruct(self, ratio, device)
        return self.evaluate_sparsity()

    def prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_random_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_random_global_struct(self, ratio, device, prune_dense_layers=False):
        _prune_random_global_struct(self, ratio, device, prune_dense_layers=False)
        return False

    def prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel'):
        _prune_magnitude_local_struct(self, ratio, device, prune_dense_layers=False, structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel'):
        _prune_magnitude_global_struct(self, ratio, device, prune_dense_layers=False,structure='kernel')
        return self.evaluate_sparsity()

    def prune_magnitude_local_unstruct(self, ratio, device, scope='layer'):
        _prune_magnitude_local_unstruct(self, ratio, device, scope='layer')
        return self.evaluate_sparsity()
        
    
    
    
    
