import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import multiprocessing
import os
import time
import numpy as np

# 设置随机种子数
seed = 8
torch.manual_seed(seed)


# start to count time
starting_time = time.time()

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        saved_path = 'C:/Users/Easyai/Desktop/gly/code/saved_models/initialization_b16.pth'
        checkpoint = {
            'state_dict': model.state_dict(),
            'best_acc': val_loss,
        }
        try:
            torch.save(checkpoint, saved_path)
            print(f'Model saved with validation loss: {val_loss}')
        except Exception as e:
            print(f"Error saving weights: {e}")


# 定义SE Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


'''
    Define part
'''
# Define a function for testing the model on the test set
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy

'''
         Data loading part
'''
# 定义数据变换,添加一些数据增强的操作，如随机翻转、随机旋转等，以增加模型的泛化能力
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
])
# 创建数据集实例
train_dataset = datasets.ImageFolder(root=r'C:/Users/Easyai/Desktop/gly/Dataset_enhanced/Training', transform=transform)
val_dataset = datasets.ImageFolder(root=r'C:/Users/Easyai/Desktop/gly/Dataset_enhanced/Validation', transform=transform)
test_dataset = datasets.ImageFolder(root=r'C:/Users/Easyai/Desktop/gly/Dataset_enhanced/Testing', transform=transform)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
        Model loading part
'''
model = torchvision.models.vgg16(pretrained=True)  # 加载torch原本的vgg16模型，设置pretrained=True，即使用预训练模型
num_fc = model.classifier[6].in_features  # 获取最后一层的输入维度
num_cls = 4
model.classifier[6] = torch.nn.Linear(num_fc, num_cls)  # 修改最后一层的输出维度，即分类数
# 对于模型的每个权重，使其不进行反向传播，即固定参数
for param in model.parameters():
    param.requires_grad = True
# 将分类器的最后层输出维度换成了num_cls，这一层需要重新学习
for param in model.classifier[6].parameters():
    param.requires_grad = True

# If there is gpu, run with gpu, else run with cpu
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available, train on CPU...')
else:
    print('CUDA is available, train on GPU...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the modified model back to the GPU if available
model.to(device)
print(model)



def main():
    '''
        Training part
     '''
    # Set initial optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Learning rate attenuation: lr becomes 1/10 every 10 epoch
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the number of training epochs
    num_epochs = 100

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=5, delta=0.0001)

    # Create lists to store training and validation history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # learning rate history
    LRs = [optimizer.param_groups[0]['lr']]


    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize running loss and accuracy for this epoch
        training_since = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0


        # Iterate over the training data
        for inputs, labels in train_loader:
            # Move the inputs and labels to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the running loss and corrects
            running_loss += loss.item() * inputs.size(0)

            # Update training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Adjust the learning rate
        scheduler.step()

        # Calculate the average loss for this epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Calculate training accuracy
        epoch_acc = correct_train / total_train

        # Print the training loss and accuracy for this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc * 100:.2f}%')

        # Calculate one training epoch time
        training_time = time.time() - training_since
        print(' Training time {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
        # Save training history for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)



        # Validation loop
        validation_since = time.time()
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_loss /= total
        val_accuracy = correct / total

        # Print the validation loss and accuracy for this epoch
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

        # Calculate one training epoch time
        validation_time = time.time() - validation_since
        print('Validation time {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))

        # Check for early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save validation history for plotting
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Learning rate history
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    # Load the best model after all epochs are done
    checkpoint = torch.load('C:/Users/Easyai/Desktop/gly/code/saved_models/initialization_b16.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Test the model on the test set
    test_accuracy = test_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    download_path = 'C:/Users/Easyai/Desktop/gly/code/checkpoint.pt'

    # 初始化EarlyStopping时设置路径为保存的权重文件路径
    early_stopping = EarlyStopping(patience=5, delta=0.001, path=download_path)

    main()

total_time = time.time() - starting_time
print('Total time {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
