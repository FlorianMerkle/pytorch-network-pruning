import torch
import torchvision

def load_imagenette():
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_path='./data/imagenette2-320/train'
    imagenette_train = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=transforms
    )
    val_path='./data/imagenette2-320/val'
    imagenette_val = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=transforms
    )

    train_loader = torch.utils.data.DataLoader(imagenette_train,
                                              batch_size=256,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(imagenette_val,
                                              batch_size=256,
                                              shuffle=True)
    return train_loader, val_loader

def load_torchvision_dataset(dataset, batchsize=512, data_augmentation=False):
    if data_augmentation == True:
        train_transforms = torchvision.transforms.Compose([
            #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.Resize(40),
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),        
        ])
    if data_augmentation == False:
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),        
        ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    if dataset == 'MNIST':
        train = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test = torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transforms, download=True)
        test = torchvision.datasets.CIFAR10('./data', train=False, transform=val_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batchsize,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batchsize,
        shuffle=True,
    )
    return train_loader, test_loader