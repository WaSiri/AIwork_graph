import glob
import os
import numpy as np
from PIL import Image
import scipy.io as io

# src_dir = './test/'
# save_dir = './fer_data/'
# print('Start... ')
# img = Image.open(src_dir+'0/01495.jpg')
# res = np.array(img, dtype='uint16')
# # save to .npy
# print('Saving data...')
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# np.save(save_dir + 'origin.npy', res)
# print('Done.')
# numpy_file = np.load(save_dir+'origin.npy')
# num = 1
# strnum = str(num)
# print(num)
# print(strnum)
# label = [num]
# io.savemat('origin.mat', {'data': numpy_file,'label':label})
#
# data=io.loadmat("./origin.mat")
# print(data)

src_dir = './test/'
save_dir = './fer_data/'
data = []
label = []
single_label = [0,0,0,0,0,0]
for i in range(7):
    stri = str(i)
    img_list = glob.glob(src_dir+stri+'/*.jpg')
    print('Start... %d' %i )
    for j in range(len(img_list)):
        preimg = img_list[j]
        img = Image.open(preimg)
        single_data = np.array(img, dtype='uint16')
        data.append(single_data)
        single_label.insert(i,1)
        label.append(single_label)
        single_label = [0, 0, 0, 0, 0, 0]

print('Saving data...')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

np.save(save_dir + 'origin.npy', data)
print('Done.')

numpy_file = np.load('./fer_data/origin.npy')

io.savemat('face_test.mat', {'data': data,'label':label})

data=io.loadmat("./face_test.mat")
print(data)