import torch
from torch import nn
import torchvision.models as models
#https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
#https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class BasicBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride,downsample) -> None:
        super().__init__()
        # downsample是nn.Squential对象，用1x1的卷积实现降维
        self.downsample = downsample
        # 第一个卷积层布长可能是2也可能是1，第二个卷积层步长一定是1
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size =3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # conv1,2操作过后，平面尺寸没有变化
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # img_channels 图像通道数
    # num_layers ResNet层数
    # block 是BasicBlock的实例
    def __init__(self, img_channels=3,num_classes=10):
        super().__init__()
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=64,kernel_size=7,stride = 2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #layer1,2,3,4都是nn.Squential对象，因此也是可调用对象
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        
        return
    def forward(self,x):
        #平面size从224到56，通道从3-64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        #从通道(dim=1)开始展开，最底层逐渐合并，总的顺序是通道顺序
        #dim=0是batch，那么合并后的tensor就是(batch,channel)，同一个通道的数据全部都压缩成1维
        # flatten也可以建立可调用对象，flatten没有参数，但可能影响参数名称从而无法接收到标准预训练参数，relu也同理
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
        
        
    def _make_layer1(self)->nn.Sequential:
        return  nn.Sequential(BasicBlock(64,64,stride=1,downsample=None),BasicBlock(64,64,stride=1,downsample=None))
    def _make_layer2(self)->nn.Sequential:
        #strade=2 相当于下采样  kernel=1进行升维 downsample=None相当于直连
        downsample = nn.Sequential(nn.Conv2d(64,128,kernel_size=1,stride = 2,padding=0,bias=False),nn.BatchNorm2d(128))
        return nn.Sequential(BasicBlock(64,128,stride=2,downsample=downsample),BasicBlock(128,128,stride=1,downsample=None))
    def _make_layer3(self)->nn.Sequential:
        downsample = nn.Sequential(nn.Conv2d(128,256,kernel_size=1,stride = 2,padding=0,bias=False),nn.BatchNorm2d(256))
        return nn.Sequential(BasicBlock(128,256,stride=2,downsample=downsample),BasicBlock(256,256,stride=1,downsample=None))
    def _make_layer4(self)->nn.Sequential:
        downsample = nn.Sequential(nn.Conv2d(256,512,kernel_size=1,stride = 2,padding=0,bias=False),nn.BatchNorm2d(512))
        return nn.Sequential(BasicBlock(256,512,stride=2,downsample=downsample),BasicBlock(512,512,stride=1,downsample=None))
    
if __name__=='__main__':
    
    # model = ResNet(3,10)
    # print(model)

    # # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    # model = ResNet(3,10)
    # print(f"Model structure: {model}\n\n")
    # # https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.htmal
    # # with torch.no_grad() 或者是 parm.detach()可以关闭梯度
    # # parameters()只返回参数本身 named_parameters()返回(name,参数本身)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Require_grad:{param.requires_grad}\n")
    #     # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    # modeld = models.resnet18(weights = models.ResNet18_Weights)
    # model = ResNet(3,10)
    # pre_state_dict = modeld.state_dict()
    # cur_state_dict = model.state_dict()
    # # for k,v in pre_state_dict.items():
    # #     if v.shape != cur_state_dict[k].shape:
    # #         continue
    # #     else:
    # #         cur_state_dict[k] = pre_state_dict[k]
    # model.load_state_dict(cur_state_dict)
    # for name,param in model.named_parameters():
    #     if name=='layer4.1.conv2.weight':
    #         print(f'{name} | {param[:1]}')


    #查看python的字典使用方法 dict.keys() dict.values() dict.items() 
    # model = ResNet(3,10)
    # cur_state_dict = model.state_dict()
    # for key in cur_state_dict.keys():
    #     print(key)
    
    model  = models.resnet18()
    model.conv1 = nn.Conv2d(3,64,3,1,1)
    print(model)