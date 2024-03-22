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


# Define the Residual Block
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super(ResNeXtBlock, self).__init__()
        mid_channels = cardinality * out_channels // 32
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# Define the ResNeXt architecture
class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], cardinality)
        self.layer2 = self.make_layer(block, 128, layers[1], cardinality, stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], cardinality, stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, cardinality, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, cardinality, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
    model = ResNeXt(ResNeXtBlock, [3, 4, 6, 3], cardinality=32, num_classes=num_classes)
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
