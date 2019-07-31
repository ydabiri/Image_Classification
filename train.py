#This program trains a deep learning model to determine the name of a flower
import matplotlib.pyplot as plt
import torch
from torchvision import models,datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

def train(data_dir,epochs,lr,devicename,model_name,hidden_units,class_indexs):
    
    #data_dir = 'C:/Users/Yaghoub/Desktop/New folder/udacity/image_classifier/flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    

    # The transforms are defined here.
    data_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),\
    transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Data are loaded here
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)
    image_valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    
    # The dataloaders are defined here.
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    datatestloaders = torch.utils.data.DataLoader(image_test_datasets, batch_size=64, shuffle=True)
    datavalidloaders = torch.utils.data.DataLoader(image_valid_datasets, batch_size=64, shuffle=True)

    #########################################    
    #model = models.vgg16(pretrained=True)
    model = getattr(models, model_name)(pretrained=True)
    model.class_to_idx = image_datasets.class_to_idx

    # Parameters are freezed so no backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),('relu', nn.ReLU()),\
    ('fc2', nn.Linear(hidden_units, 102)),('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier


    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    device = torch.device("cuda:0" if devicename=="gpu" else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)

    for epoch in range(epochs):
        for inputs, labels in dataloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in datavalidloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(datatestloaders):.3f}.. "
                      f"Test accuracy: {accuracy/len(datatestloaders):.3f}")
                running_loss = 0
                model.train()
                
    model.class_to_idx = image_datasets.class_to_idx
    model.epochs = epochs
    return model



