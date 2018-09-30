#########################
# Neural Style Transfer #
#########################
'''
VGG19
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): ReLU(inplace)
(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(6): ReLU(inplace)
(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(8): ReLU(inplace)
(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(11): ReLU(inplace)
(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(13): ReLU(inplace)
(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(15): ReLU(inplace)
(16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(17): ReLU(inplace)
(18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(20): ReLU(inplace)
(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(22): ReLU(inplace)
(23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(24): ReLU(inplace)
(25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(26): ReLU(inplace)
(27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(29): ReLU(inplace)
(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(31): ReLU(inplace)
(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(33): ReLU(inplace)
(34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(35): ReLU(inplace)
(36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

The Neural Style Transfer includes only 10 layers of VGG19.
Conv: ['0','2','5','7', '10']
ReLU: ['1','3','6','8']
Maxpool: ['4','9']
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

import argparse

plt.ion()

## import target images(content image and style image)
def loader(content_img, style_img, size):
    content_img = Image.open(content_img)
    style_img = Image.open(style_img)
    transform = transforms.Compose([
        transforms.Resize((size,size)),  # scale imported image
        transforms.ToTensor()  # the order is important 'Resize first and ToTensor' 
    ])
    content_img_rsz = transform(content_img)
    style_img_rsz = transform(style_img)

    imshow(content_img_rsz, title = 'Content Image')
    imshow(style_img_rsz, title = 'Style Image')

    content_img = content_img_rsz.unsqueeze(0)
    style_img = style_img_rsz.unsqueeze(0)

    return content_img, style_img


## visualize resized target images
def imshow(image, title = None):
    image = image.clone()
    image = image.squeeze(0)
    pil = transforms.ToPILImage()
    target_img_PIL = pil(image)
    plt.figure()
    plt.imshow(target_img_PIL)
    if title is not None:
        plt.title(title)
    plt.pause(1)


## gram_matrix for the 'style loss function'
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(-1, h*w)
    G = torch.mm(features, features.t())
    return G.div(b*c*h*w)

## 'content loss function' and 'style loss function'
class CL(nn.Module):
    def __init__(self, target):
        super(CL, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class SL(nn.Module):
    def __init__(self, target):
        super(SL, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


## Generating the 'Neural Style Transfer' Model
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def nst_model(content_img, style_img):
    vgg = models.vgg19(pretrained=True).features.eval()
    normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    content_img = content_img.detach()
    style_img = style_img.detach()

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for name, layer in vgg._modules.items():
        if name in ['0','2','5','10']:
            model.add_module('conv_{}'.format(i),layer)

            style_target = model(style_img)
            style_loss = SL(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i),style_loss)
            
            i += 1

        elif name in ['7']:
            model.add_module('conv_{}'.format(i),layer)

            content_target = model(content_img)
            content_loss = CL(content_target)
            content_losses.append(content_loss)
            model.add_module('contentloss_{}'.format(i),content_loss)
            style_target = model(style_img)
            style_loss = SL(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i),style_loss)

            i += 1

        elif name in ['1','3','6','8']:
            layer = nn.ReLU(inplace=False)
            model.add_module('relu_{}'.format(i),layer)

            i += 1

        elif name in ['4','9']:
            model.add_module('maxpool_{}'.format(i),layer)
            
            i += 1

        elif name == '11':
            break

    return model, style_losses, content_losses


def main(args):

    content_img = args.content_img
    style_img = args.style_img
    size = args.size
    steps = args.steps
    c_weight = args.c_weight
    s_weight = args.s_weight

    content_img, style_img = loader(content_img, style_img, size = size)
    input_img = content_img.clone()
    

    model, style_losses, content_losses  = nst_model(content_img, style_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    step = [0]
    while step[0] <= steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            output = model(input_img)

            cl = 0
            sl = 0

            for c_loss in content_losses:
                cl += c_loss.loss * c_weight
            for s_loss in style_losses:
                sl += s_loss.loss * s_weight

            loss = cl + sl
            loss.backward()

            if step[0] % 50 == 0:
                print('Step : {}'. format(step))
                print('Style Loss : {:3f} Content Loss: {:3f}'.format(
                    sl.item(), cl.item()))

            step[0] += 1

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0,1)
    return input_img

    imshow(input_img, title = 'Input image')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img', type=str, default = 'images/dancing.jpg')
    parser.add_argument('--style_img', type=str, default = 'images/picasso.jpg')
    parser.add_argument('--size', type=int, default = 128)
    parser.add_argument('--steps', type=int, default = 300)
    parser.add_argument('--c_weight', type=int, default = 1, help='weighting factor for content reconstruction')
    parser.add_argument('--s_weight', type=int, default = 1000000, help='weighting factor for style reconstruction')

    args = parser.parse_args()
    print(args)
    output = main(args)

    plt.figure()
    imshow(output, title='Output Image')
    plt.pause(5)

    plt.ioff()
    plt.show()













