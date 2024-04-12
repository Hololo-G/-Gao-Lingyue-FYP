import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import win32clipboard
import torch
import torchvision
import os
from torchvision import transforms
import io
from torchvision.models import vgg16

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

model.eval()

# 数据集目录路径
dataset_dir = 'D:/SBC/FYP/Dataset_corrected/Testing/'

# 获取数据集目录下的所有子目录（即类别名称）
class_labels = os.listdir(dataset_dir)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()
    return predicted_class_index

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        display_image_and_predict(file_path)

def paste_image():
    win32clipboard.OpenClipboard()
    if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
        data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
        image = Image.open(io.BytesIO(data))
        # 保存黏贴的图像到临时文件
        temp_path = 'temp_image.png'
        image.save(temp_path)
        win32clipboard.CloseClipboard()
        display_image_and_predict(temp_path)
        # 删除临时文件
        os.remove(temp_path)

def display_image_and_predict(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 300))  # 缩放图像
    imgtk = ImageTk.PhotoImage(image)
    label_image.configure(image=imgtk)
    label_image.image = imgtk  # 保存图片对象的引用
    predicted_class_index = predict(image_path)
    predicted_class_label = class_labels[predicted_class_index]
    label_result.config(text=f"Predicted brain tumor result: {predicted_class_label}")

# 创建主窗口
root = tk.Tk()
root.title("Brain Tumor Classification Tool")

# 创建图像显示面板
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# 创建标签显示图像
label_image = tk.Label(root)
label_image.pack()

# 创建标签显示预测结果
label_result = tk.Label(root, text="")
label_result.pack()

# 添加一个按钮，用于从剪贴板粘贴图像
paste_button = tk.Button(root, text="Paste Image", command=paste_image)
paste_button.pack()

# 添加一个按钮，用于选择文件进行预测
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

root.mainloop()