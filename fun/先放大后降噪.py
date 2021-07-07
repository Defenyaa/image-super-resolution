import numpy as np
from PIL import Image
from ISR.models import RDN,RRDN
img = Image.open('../data/input/test_images/1.jpg')
lr_img = np.array(img)



# 对抗神经网络

rdn = RDN(weights='psnr-small')
sr_img = rdn.predict(lr_img)
print("放大完毕")


rdn = RDN(weights='noise-cancel')
ssr_img = rdn.predict(sr_img)
print("降噪完毕")

i = Image.fromarray(ssr_img)
Image._show(i)

print("end")