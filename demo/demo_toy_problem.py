from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)


# In[3]:
import sys

sys.path.append('.')

from tf_unet import image_gen

from tf_unet import unet, util, image_util




data_provider = image_util.ImageDataProvider("/scratch/haoran/transform/data/all/*.tif")


net = unet.Unet(channels=3, n_class=2, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider, "./unet_trained", training_iters=64, epochs=20, display_step=2)


batch_num = 10
for i in range(batch_num):
    batch_size = 4
    x_test, y_test = data_provider(batch_size)
    #print(x_test.shape)
    #print(y_test)
    

    #[n, nx, ny, channels]

    prediction = net.predict("./unet_trained/model.ckpt", x_test)
    '''print(prediction)
    for row in prediction[3]:
        for column in row:
            if column[1] > 0.1:
                print(column)'''

    for j in range(batch_size):
        #print(x_test[j])
        im_x = Image.fromarray((x_test[j]*255).astype(np.uint8))
        #im_y = Image.fromarray((y_test[j,:,:,1])*256)
        #im_p = Image.fromarray((prediction[j,:,:,1]*256).astype(np.uint8))
        
        im_x.save('x'+str(i)+' '+str(j)+ ".jpg")
        #im_y.save('y'+str(i)+' '+str(j)+ ".jpg")
        #im_p.save('p'+str(i)+' '+str(j)+ ".jpg")
        cv2.imwrite('y'+str(i)+' '+str(j)+ ".jpg", ((y_test[j,:,:,1])*255).astype(np.uint8))
        cv2.imwrite('p'+str(i)+' '+str(j)+ ".jpg", (prediction[j,:,:,1]*255).astype(np.uint8))
