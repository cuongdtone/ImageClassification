import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import os
import argparse
import matplotlib.pyplot as plt
import pickle
from utils.evaluate import evaluate_cm
from utils.load_model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import custom model 
from models.MobileNetV3 import mobilenet_v3_small
model = mobilenet_v3_small(pretrained=True, num_classes=2).to(device)

# Get parameter
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path dataset')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--project', default='runs', help='path save model')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--img-size', type=int, default=224)
args = vars(parser.parse_args())


input_path = args['dataset']
epochs = args['epochs']
batch_size = args['batch_size']
input_size = args['img_size']
lr = 0.001
path_save_model = args['project']
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# Don't touch here

def train_model(model, criterion, optimizer, num_epochs=3):
    best_loss_val = -1
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        last_time = time.time()
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            print("Memory: ")
            for inputs, labels in dataloaders[phase]:
                total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                print ("\033[A                             \033[A")
                print("    Memory: %0.2f GB / %0.2f GB"%(used_memory_before/1024, total_memory/1024))
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #print(preds)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            if phase =="train":
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            print('    {} loss: {:.4f}, accuracy: {:.4f}'.format(phase.title(), epoch_loss, epoch_acc))
        print("    Epoch Time: ", str(datetime.timedelta(seconds=time.time()-last_time)))
        if best_loss_val == -1:
            best_loss_val = running_loss
        if best_loss_val>=running_loss:
            best_loss_val = running_loss
            torch.save(model, path_save_model + '/best.pt')
            print("    Saved best model at epoch ", epoch+1)
        print('-' * 10)
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(range(len(train_acc)), train_acc, "r", label="Train Accuracy")
    plt.plot(range(len(val_acc)), val_acc, 'g', label="Val Accuracy")
    plt.xlabel("epoch")
    plt.title("Accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.subplot(1,2,2)
    plt.plot(range(len(train_loss)), train_loss, "r", label="Train Loss")
    plt.plot(range(len(val_loss)), val_loss, 'g', label="Val Loss")
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.legend(loc="best")
    plt.grid()
    fig.savefig(path_save_model + '/result.png', dpi=500)
    return model

if not os.path.exists(path_save_model):
    os.mkdir(path_save_model)
temp = 0
while os.path.exists(path_save_model + '/exp%d'%(temp)):
    temp = temp+1
path_save_model = path_save_model + '/exp%d'%(temp)
os.mkdir(path_save_model)


with open(path_save_model + '/parameter.txt', 'w') as file:
    temp = "inputsize = %d"%(input_size)
    temp = temp + '\nepochs = %d'%(epochs)
    temp = temp + '\nbatchsize = %d'%(batch_size)
    file.write(temp)
    file.close()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_datasets = {
    'train':
    datasets.ImageFolder(input_path + '/train',
    transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    'validation':
    datasets.ImageFolder(input_path + '/valid',
    transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]))}

file = open(path_save_model + '/label.txt', "w")
class_id = image_datasets['train'].class_to_idx
for key, value in class_id.items():
    file.write(key +" : "+ str(value)+ '\n')
file.close()

train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=batch_size, shuffle=True,
    num_workers=batch_size, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    image_datasets['validation'],
    batch_size=batch_size, shuffle=False,
    num_workers=batch_size, pin_memory=True)

dataloaders = {
    'train':
    train_loader,
    'validation':
    val_loader
}
if __name__=='__main__':
    model_trained = train_model(model, criterion, optimizer, num_epochs=epochs)

    #model = Model(path_save_model)
    #evaluate_cm(model, input_path+ '/test', save_cm=path_save_model, normalize=False, show=False)

    torch.save(model_trained, path_save_model + '/last.h5')
    print('Saved last model at ', path_save_model, "/last.h5")
    print("Complete !")
