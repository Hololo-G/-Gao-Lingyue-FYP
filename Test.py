import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define SE Block
class SEBlock(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = datasets.ImageFolder(root=r'D:/SBC/FYP/Dataset_enhanced/Testing', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Load model
model = torchvision.models.vgg16(pretrained=True)
num_fc = model.classifier[6].in_features
num_cls = 4
model.classifier[6] = torch.nn.Linear(num_fc, num_cls)

# Copy parameters and add SE Block
for name, module in model.named_children():
    if name == 'features':
        new_features = torch.nn.Sequential()
        for n, m in module.named_children():
            new_features.add_module(n, m)
            if isinstance(m, torch.nn.Conv2d) and n == '7':
                in_channels = m.out_channels
                se_block = SEBlock(in_channels)
                new_features.add_module(n + '_se', se_block)
        setattr(model, name, new_features)
    else:
        setattr(model, name, module)

# Move model to device
model.to(device)

# Load the saved weights
checkpoint = torch.load('D:/SBC/FYP/Final/saved_models/initialization_b16_se.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])


# Define a function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    class_accuracy = []
    class_precision = []
    class_recall = []
    class_f1 = []
    for i in range(cm.shape[0]):
        class_accuracy.append(cm[i, i] / np.sum(cm[i, :]))
        class_precision.append(cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) != 0 else 0)
        class_recall.append(cm[i, i] / np.sum(cm[i, :]))
        class_f1.append(2 * class_precision[-1] * class_recall[-1] / (class_precision[-1] + class_recall[-1]) if (
                                                                                                                             class_precision[
                                                                                                                                 -1] +
                                                                                                                             class_recall[
                                                                                                                                 -1]) != 0 else 0)

    return cm, accuracy, precision, recall, f1, class_accuracy, class_precision, class_recall, class_f1


# Evaluate the model
cm, accuracy, precision, recall, f1, class_accuracy, class_precision, class_recall, class_f1 = evaluate_model(model,
                                                                                                              test_loader)

# Print the results
print("Confusion Matrix:")
print(cm)
print("\nOverall Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClass-wise Metrics:")
for i, cls in enumerate(test_dataset.classes):
    print(f"Class: {cls}")
    print(f"\tAccuracy: {class_accuracy[i]}")
    print(f"\tPrecision: {class_precision[i]}")
    print(f"\tRecall: {class_recall[i]}")
    print(f"\tF1 Score: {class_f1[i]}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()