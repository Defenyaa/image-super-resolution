import numpy as np
from PIL import Image
from ISR.models import RDN,RRDN

img = Image.open('data/input/test_images/1.jpg')
lr_img = np.array(img)



# 对抗神经网络
# rdn = RRDN(weights='gans')
# rdn = RDN(weights='psnr-small')
# rdn = RDN(weights='psnr-large')
rdn = RDN(weights='noise-cancel')

# sr_img = rdn.predict(lr_img)

# by_patch_of_size=50 对大照片处理
sr_img = rdn.predict(lr_img, by_patch_of_size=50)


i = Image.fromarray(sr_img)

Image._show(i)

print("end")