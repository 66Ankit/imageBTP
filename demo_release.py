
import argparse
import matplotlib.pyplot as plt
import cv2
import  os
from PIL import Image

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='op/image32.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

path="demo"
colorizer_eccv16 = eccv16(pretrained=True).eval()
for name in os.listdir(path):

	img_name=os.path.join(path,name)

	im=cv2.imread(img_name)
	if(opt.use_gpu):
		colorizer_eccv16.cuda()
	img = load_img(img_name)

	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()
	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
	print(out_img_eccv16.shape)
	print(out_img_eccv16[0][0][0])
	print(type(out_img_eccv16))
	im = Image.fromarray(np.uint8(out_img_eccv16*255),'RGB')
	im.save("demo_out/" + name)
	print(type(im))
	plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)