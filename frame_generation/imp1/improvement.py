import numpy as np
import os
from PIL import Image



x_test = []

x_test1 = []
k = 1
directory = 'C:\\Users\\raman\\OneDrive\\Desktop\\test'
p=0
for filename in os.listdir(directory):
    if (k == 500):
        break
    elif (k == 499):
        image = Image.open(os.path.join(directory, filename))
        image = image.resize((512, 512))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_array = (image_array / 255.0).astype('float32')
        x_test1.append(image_array)
    else:
        image = Image.open(os.path.join(directory, filename))
        image = image.resize((512, 512))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_array = (image_array / 255.0).astype('float32')
        if p == 0:
            x_test.append(image_array)
            p = p + 1
        if (p == 1):
            x_test1.append(image_array)
            x_test.append(image_array)
    k=k+1