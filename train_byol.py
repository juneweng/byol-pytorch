import torch
from byol_pytorch import BYOL
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable


use_gpu = torch.cuda.is_available()


resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256
    # ,
    # hidden_layer = 'avgpool'
)

learner=learner.cuda()
# if torch.cuda.is_available:   
#     learner=nn.DataParallel(learner,device_ids=[0,1,2]) # multi-GPU

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)


# def sample_unlabelled_images():
#     return torch.randn(20, 3, 256, 256)

for epoch in range(1000):
    f = open('pytorch_out.txt', 'a')
    for batch_id, data in enumerate(trainloader):
        images, _ = data 
        images.permute(0,3,1,2)
        if use_gpu:
            images = Variable(images.cuda())
        # print(images.shape)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        if batch_id % 50 == 0:
            item = "[Epoch %d, batch %d] loss: %.5f" % (epoch, batch_id, loss)
            print(item)
            f.write(str(item)+'\n')
    f.close()            


# save your improved network
torch.save(resnet.state_dict(), './improved-net.txt')