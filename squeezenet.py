import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader


class FeaturesInputLoad():
    def __init__(self, PATH):
        self.PATH = PATH
    def preprocessing(self):
        image_torch_list = []
        for image in glob.glob("{}/*".format(self.PATH)):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
            image_resize = cv2.resize(image, (244, 244), interpolation = cv2.INTER_AREA)
            image_normalized = cv2.normalize(image_resize, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image_torch = torch.from_numpy(image_normalized).permute(2, 0, 1)
            image_torch_list.append(image_torch)

        concatenated_tensors = []
        for tensor in image_torch_list:
            concatenated_tensors.append(tensor.unsqueeze(0))
        result_image_tensor = torch.cat(concatenated_tensors, dim=0)

        return result_image_tensor


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256)
        )
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    # features = torch.randn((28,3,224,224))
    loadinput = FeaturesInputLoad("C:/Users/Phongsakhon/Projects/Image-CS/source/data/Humans")
    features = loadinput.preprocessing()
    labels = ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "B", "B", "B",
                "B", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C", "C", "C", "C", "C"]
    label_mapping = {"A": 0, "B": 1, "C": 2}
    numeric_labels = torch.tensor([label_mapping[label] for label in labels], dtype=torch.int64)
    dataset = TensorDataset(features, numeric_labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    num_classes = 3
    model = SqueezeNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

    num_epochs = 50 # Adjust the number of epochs as needed

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    image = cv2.imread("C:/Users/Phongsakhon/Projects/Image-CS/source/data/Humans/27.png", cv2.IMREAD_COLOR)
    image_resize = cv2.resize(image, (244, 244), interpolation = cv2.INTER_AREA)
    image_normalized = cv2.normalize(image_resize, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_torch = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    prediction = model(image_torch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()

    print(prediction)
    print(class_id)
    print(score)
