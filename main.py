import torch
import torchvision
import compressai
from model import Compressor
from loss import RateDistortionLoss
from utils import concat_images
from tqdm import *

device = "cuda:0"
model = Compressor()
loss_fn = RateDistortionLoss(0.01)
optim = torch.optim.AdamW(model.parameters(),lr=1e-4)
# train_data = compressai.datasets.ImageFolder('./data',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((768, 768)), 
#                                                                                                 torchvision.transforms.ToTensor()]),split='train')
train_data = torchvision.datasets.CIFAR10('./data',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((768, 768)), 
                                                                                                torchvision.transforms.ToTensor()]),download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

model.to(device)
model.train()
start_epoch=0
nepoch = 1
for epoch in range(start_epoch+1, nepoch+1):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_id, (data,_) in loop:
        data = data.to(device)
        result = model(data)
        x_hat = result['x_hat']
        y_likelihood = result['likelihoods']['y']
        z_likelihood = result['likelihoods']['z']
        loss = loss_fn(data,x_hat,y_likelihood,z_likelihood)

        loss.backward()
        optim.step()
        optim.zero_grad()

        loop.set_description(f'Epoch [{epoch}/{nepoch}]')
        loop.set_postfix(loss = loss.item())

    if epoch % 50 == 0:
        reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat[0].to('cpu'))
        image = torchvision.transforms.ToPILImage(mode='RGB')(data[0].to('cpu'))
        result_image = concat_images(image, reconstructed_image)
        result_image.save("train_images/epoch{}batch{}.png".format(epoch, batch_id))