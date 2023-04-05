import torch
import torchvision
import compressai
from model import Compressor
from light_model import Light_Compressor
from checkerboard import Cheng2020AnchorwithCheckerboard
from loss import RateDistortionLoss
from utils import concat_images
from tqdm import *
from torch.hub import load_state_dict_from_url
from compressai.zoo.pretrained import load_pretrained
from compressai.losses.rate_distortion import RateDistortionLoss as rdloss

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = "cuda:0"
pretrain = False
# model = Compressor(192, 320)
# model = Light_Compressor(192, 320)
model = Cheng2020AnchorwithCheckerboard(192)
if pretrain:
    state_dict = load_state_dict_from_url('https://compressai.s3.amazonaws.com/models/v1/mbt2018-5-b4a046dd.pth.tar', progress=True)
    state_dict = load_pretrained(state_dict)
    N = state_dict["g_a.0.weight"].size(0)
    M = state_dict["g_a.6.weight"].size(0)
    # model = compressai.zoo.image.mbt2018(quality=5,pretrained=True)
    model = model.from_state_dict(state_dict)

loss_fn = rdloss(lmbda=0.01,metric='mse')
# loss_fn = RateDistortionLoss(0.1)
optim = torch.optim.AdamW(model.parameters(),lr=2e-4)
# train_data = compressai.datasets.ImageFolder('./data',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((768, 768)), 
#                                                                                                 torchvision.transforms.ToTensor()]),split='train')
train_data = torchvision.datasets.CIFAR10('./data',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((192, 192)), 
                                                                                                torchvision.transforms.ToTensor()]),download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
# test 
# model.to(device)
# model.eval()
# start_epoch=0
# nepoch = 1
# with torch.no_grad():
#     for epoch in range(start_epoch+1, nepoch+1):
#         loop = tqdm(enumerate(train_loader), total=len(train_loader))
#         for batch_id, (data,_) in loop:
#             data = data.to(device)
#             result = model(data)
#             x_hat = result['x_hat']
#             y_likelihood = result['likelihoods']['y']
#             z_likelihood = result['likelihoods']['z']

#             reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat[0].to('cpu'))
#             image = torchvision.transforms.ToPILImage(mode='RGB')(data[0].to('cpu'))
#             result_image = concat_images(image, reconstructed_image)
#             result_image.save("demo_images/epoch{}batch{}.png".format(epoch, batch_id))
            
#             if batch_id > 20:
#                 break
# train
model.to(device)
model.train()
start_epoch=0
nepoch = 4000
for epoch in range(start_epoch+1, nepoch+1):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_id, (data, _) in loop:
        data = data.to(device)
        result = model(data)

        # x_hat = result['x_hat']
        # likelihood = result['likelihoods']
        loss = loss_fn(result, data)

        loss['loss'].backward()
        optim.step()
        optim.zero_grad()

        loop.set_description(f'Epoch [{epoch}/{nepoch}]')
        loop.set_postfix(loss = loss['loss'].item(), mse=loss['mse_loss'].item(),bpp=loss['bpp_loss'].item())
        if batch_id % 20 == 0:
            reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(result['x_hat'][0].to('cpu'))
            image = torchvision.transforms.ToPILImage(mode='RGB')(data[0].to('cpu'))
            result_image = concat_images(image, reconstructed_image)
            result_image.save("train_images/epoch{}batch{}.png".format(epoch, batch_id))

# from matplotlib import pyplot as plt
# plt.plot(list(range(len(loss_list))), loss_list)
# plt.savefig('loss.png')