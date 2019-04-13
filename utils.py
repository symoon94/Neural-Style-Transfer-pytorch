import torch
import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt


## import target images(content image and style image)
def loader(content_img, style_img, size):
    content_img = Image.open(content_img)
    style_img = Image.open(style_img)
    content_img = convert_mode(content_img)
    style_img = convert_mode(style_img)
    transform = transforms.Compose([    # convert img into the data format
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


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

## RGBA 2 RGB
def convert_mode(img):
    if img.mode == "RGB":
        return img
    elif img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.save(img.filename, quality=100)
        converted_img = Image.open(img.filename)
        return converted_img

## check if images are successfully loaded
# loader("images/arizona.jpg", "images/snow.jpg", 128)