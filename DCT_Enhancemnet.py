import numpy as np
from numpy.core.shape_base import block
import cv2
from functions import *

class BlockSlicing():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def slicing(self, x):
        h, w = np.shape(x)
        if np.mod(w, self.width) == 0 and np.mod(h, self.height) == 0:
            blocks = np.reshape(np.ravel(x, order='F'), (self.width, h, -1), order='F')
            blocks = np.reshape(blocks, (self.width, self.height, -1))
            return blocks
        else:
            print("Unmatch image size and block size")
            return None

    def unslicing(self, blocks, w, h):
        if np.mod(w, self.width) == 0 and np.mod(h, self.height) == 0:
            y = np.reshape(blocks, (self.width, h, -1))
            y = np.reshape(np.ravel(y, order='F'), (h, w), order='F')
            return y
        else:
            print("Unmatch image size and block size")
            return None            


# class Enhancement():
#     def __init__(self, strength, alpha, block_size):
#         self.strength = strength
#         self.alpha = alpha
#         self.block_size = block_size


block_size = 128
in_bgr = cv2.imread("Boat.png")
in_bgr = cv2.resize(in_bgr,(block_size*4, (block_size)*3))

h, w, c = np.shape(in_bgr)
h = np.int(np.floor(h/block_size) * block_size)
w = np.int(np.floor(w/block_size) * block_size)
bgr = in_bgr[:h,:w,:]
bgr = in_bgr

yuv = bgr2yuv(bgr)
Y = np.array(yuv[:,:,0])

slicer = BlockSlicing(block_size, block_size)

# print(np.shape(Y), np.shape(np.ravel(Y, order='F')))
# blocks = np.reshape(np.ravel(Y, order='F'), (block_size,h,-1), order='F')
# print(np.shape(blocks))

# blocks = np.reshape(blocks, (block_size,block_size,-1))
# print(np.shape(blocks))

blocks = slicer.slicing(Y)
n_blocks = np.shape(blocks)[2]
print(n_blocks)

Ydct = np.empty([block_size, block_size, 0])
blocks_out = np.empty([block_size, block_size, 0])
# for i in range(n_blocks):
#     # Ydct.append(cv2.dct(blocks[:,:,i]))
#     # print(i, np.shape(cv2.dct(blocks[:,:,i])))

#     block_dct = cv2.dct(blocks[:,:,i])
#     # print(np.shape(Ydct), np.shape(block_dct[..., np.newaxis]))
#     Ydct = np.concatenate((Ydct, block_dct[..., np.newaxis]), axis=2)

# for i in range(n_blocks):
#     block_idct = cv2.idct(Ydct[:,:,i])
#     blocks_out = np.concatenate((blocks_out, block_idct[..., np.newaxis]), axis=2)

#     # blocks_out.append(cv2.idct(Ydct[:,:,i]))
    

# print('diff : ', np.sum(np.abs(blocks - blocks_out)))

for i in range(n_blocks):
    cv2.imshow("block", blocks[:,:,i]/255)
    cv2.waitKey(0)



Y_out = slicer.unslicing(blocks, w, h)

# blocks_out = np.reshape(blocks, (block_size, h,-1))
# Y_out      = np.reshape(np.ravel(blocks_out, order='F'), (h, w), order='F')

# Y_out = blocks_out.reshape(h,w)

cv2.imshow("Y", Y/255)
# cv2.imshow("Y_out", Y_out/255)
cv2.waitKey(0)

# yuv[:,:,0] = np.uint8(Y_out)
# bgr2 = yuv2bgr(yuv)
# out_bgr = np.asanyarray(in_bgr)
# out_bgr[:h,:w,:] = bgr2
# cv2.imwrite("Enh_out.png", out_bgr)