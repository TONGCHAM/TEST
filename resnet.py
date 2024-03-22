import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


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


# Define the basic block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the Bottleneck block for ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive Avg Pooling
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # Flatten the output for linear layer
        out = self.linear(out)
        return out

# Define the ResNet models
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


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
    model = ResNet50(num_classes)
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
