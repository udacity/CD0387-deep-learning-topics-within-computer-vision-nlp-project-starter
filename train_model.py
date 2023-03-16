#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import sys 
import os

import argparse
import logging

# from smdebug import modes
# from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(model, train_loader, optimizer, loss_criterion, epoch):
    """Train the model for one epoch
    
    Arguments:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer used for training
        loss_criterion: Loss function
        epoch: Current epoch number (int)
    """

    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Sets all gradients in the optimizer to 0 before starting a forward pass. This is done to prevent any accumulated gradients from affecting the current forward pass.
        optimizer.zero_grad()
        
        # Get model output for current batch
        output = model(data)
        
        # Compute loss
        loss = loss_criterion(output, target)
        
        # Compute gradients with respect to the model's parameters
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Print loss statistics every 100 batches
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )



def test(model, test_loader, loss_criterion):
    """Evaluate the model on the test set
    
    Arguments:
        model: PyTorch model
        test_loader: DataLoader for test data
        loss_criterion: Loss function
        
    Returns:
        test_loss: Average test loss
        accuracy: Test accuracy in percentage
    """
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Get model output for current batch
            output = model(data)
            
            # Accumulate batch loss
            test_loss += loss_criterion(output, target).item()  
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)  
            
            # Count number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Compute accuracy
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # Compute average test loss
    test_loss /= len(test_loader.dataset)
    
    # Print test statistics
    print(
        "\Val set: Average loss: {:.4f}, Accuracy: {:.4f}% \n".format(
            test_loss, accuracy
        )
    )
    
    
def get_model(model_name):
    '''
    Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.__dict__[model_name](pretrained=True)
    
    #  When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False
    for param in model.parameters():
        param.requires_grad = False
        
    # We also need to add a fully connected layer on top of it which we do use the Sequential API.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    return model


def create_data_loaders(data_dir, batch_size, val_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # Image Augmentation
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=f"{data_dir}/train", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    validset = torchvision.datasets.CIFAR10(
        root=f"{data_dir}/test", train=False, download=True, transform=transform_valid
    )
    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=val_batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def model_fn(model_dir):
    logger.info("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = get_model('resnet50')
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
    
def main(args):
    '''
    Initialize a model by calling the net function
    '''
    model = get_model(args.model_name)
    
    '''
    Create profiling hook and register loss
    '''
    hook = None # get_hook(create_if_not_exists=True) # Commented temp 
    
    '''
    Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    if hook:
        hook.register_loss(loss_criterion)
    
    '''
    Get dataset loaders
    '''
    train_loader, val_loader = create_data_loaders(args.data_dir, args.batch_size, args.val_batch_size)
    
    for epoch in range(1, args.epochs + 1):
        '''
        Call the train function to start training your model
        '''
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        train(model, train_loader, optimizer, loss_criterion, epoch)
    
        '''
        Test the model to see its accuracy
        '''
        if hook:
            hook.set_mode(modes.EVAL)
        model.eval()
        test(model, val_loader, loss_criterion)
    
    '''
    Save the trained model
    '''
    save_model(model, args.model_dir)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    
    # Hyper parameters
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.5)")
    
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''
    args = parser.parse_args()
    
    main(args)
