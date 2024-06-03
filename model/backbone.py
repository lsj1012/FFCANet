import torch,pdb
import torchvision
import torch.nn.modules

# 该模块本质上是具有批量归一化 （vgg16_bn） 的 VGG16 模型的修改版本。让我们分解一下代码：vgg16bn
# class vgg16bn(torch.nn.Module):：此行定义了一个名为 的类，该类继承自 。vgg16bn torch.nn.Module
# def __init__(self, pretrained=False):：这是类的构造函数方法。它初始化对象。它采用一个可选参数，该参数默认为 ，
# 指示是否为VGG16_bn模型加载预训练权重。pretrainedFalse
# super(vgg16bn, self).__init__()：此行调用父类 （） 的构造函数以正确初始化对象。torch.nn.Module
# model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())：
# 此行使用指定的参数从 torchvision 加载VGG16_bn模型并检索其特征。它将要素转换为列表。pretrained
# model = model[:33] + model[34:43]：此行修改了VGG16_bn模型的体系结构。它删除了最后一个最大池化层（索引 33）和随后的全连接层（索引 34 到 42）。
# 这有效地删除了VGG16_bn模型的分类器部分。
# self.model = torch.nn.Sequential(*model)：此行使用修改后的图层列表 （） 创建模块。
# 此顺序模块表示修改后的VGG16_bn模型架构。torch.nn.Sequential model
# def forward(self, x):：此方法定义模块的前向传递。
# return self.model(x)：此行将修改后的VGG16_bn模型应用于输入并返回输出。x
# 总体而言，此代码定义了一个没有完全连接层的自定义VGG16_bn模型，使其适用于特征提取或作为迁移学习任务中的特征提取器。
class vgg16bn(torch.nn.Module):
    def __init__(self, pretrained = False):
        super(vgg16bn, self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)



# 该模块允许您根据指定的层数创建 ResNet 架构的不同变体（ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152、
# ResNeXt-50-32x4d、ResNeXt-101-32x8d、Wide ResNet-50-2、Wide ResNet-101-2）。resnet
# 让我们分解一下代码：
# class resnet(torch.nn.Module):：此行定义了一个名为 的类，该类继承自 。resnet torch.nn.Module
# def __init__(self, layers, pretrained=False):：这是类的构造函数方法。它初始化对象。它需要两个参数：
# layers：指示 ResNet 架构中层数的字符串（例如，'18'、'34'、'50'、'101'、'152'、'50next'、'101next'、'50wide'、'101wide'）。
# pretrained：指示是否为 ResNet 模型加载预训练权重的布尔值。此参数默认为 。False
# 在构造函数内部，根据 的值，它使用 module 实例化相应的 ResNet 变体。layers torchvision.models
# 然后，将 ResNet 模型的组件（卷积层、批量归一化、激活函数、最大池化和每个残差层）分配给自定义模块的属性（、、）。
# self.conv1 self.bn1 self.relu self.maxpool self.layer1 self.layer2 self.layer3 self.layer4
# def forward(self, x):：此方法定义模块的前向传递。
# 在正向方法中，它执行通过 ResNet 模型的层进行前向传递，返回每个阶段 （， ， ） 之后的激活，分别对应于第二、第三和第四残差阶段的输出。x2x3x4
# 总体而言，此类提供了一种灵活的方法来实例化 ResNet 架构的不同变体并从网络的不同阶段提取特征。resnet
class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1 #卷积层
        self.bn1 = model.bn1 #批量归一化
        self.relu = model.relu #激活函数
        self.maxpool = model.maxpool #最大池化层
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4
