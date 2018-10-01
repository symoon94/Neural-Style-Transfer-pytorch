import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--content_img', type=str, default = 'images/me3.jpg')
parser.add_argument('--style_img', type=str, default = 'images/chagall2.jpg')
parser.add_argument('--size', type=int, default = 128)
parser.add_argument('--steps', type=int, default = 300)
parser.add_argument('--c_weight', type=int, default = 1, help='weighting factor for content reconstruction')
parser.add_argument('--s_weight', type=int, default = 100000, help='weighting factor for style reconstruction')

args = parser.parse_args()
print(args)