import torch
from torch import nn
from resnet18 import ResNet
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet()
model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
model.load_state_dict(torch.load('checkpoint/epoch030resnet18_cifar10.pt',map_location=torch.device(device)))
model = model.to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
test_data = datasets.CIFAR10('.',train=False,transform=transform,download=True)
test_loader = DataLoader(test_data,16)
total_correct = 0
model.eval()

for i,(batch,labels) in enumerate(test_loader):
    batch = batch.to(device)
    labels = labels.to(device)
    output = model(batch).to(device)
    res = torch.argmax(output,1)
    correct = (res==labels).sum().item()
    total_correct += correct
    if i%100==0:
        print(f'cur test batch is No{i}')
        
print(f'accuracy is {100*total_correct/10000}%')








