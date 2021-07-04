import numpy as np
from numpy.core.shape_base import block
import cv2
import math
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


class CSF():
    def __init__(self, size, Rvd, Pic, s, r, params, Lmax=256, Lmin=0):
        self.size = size
        self.Rvd = Rvd
        self.Pic = Pic
        self.s = s
        self.r = r
        self.params = params
        self.Lmax = Lmax
        self.Lmin = Lmin
        self.G = self.Lmax# - self.Lmin + 1

        self.theta = 2*math.atan(1/2/self.Pic/self.Rvd)
        self.s_pi_v = np.ones([self.size, 1])*math.sqrt(2/self.size)
        self.s_pi_v[0] = math.sqrt(1/self.size)
        self.s_pi_h = self.s_pi_v

        self.Tbase = np.empty([self.size, self.size])

    def makeTbase(self):
        a, b, c, d = self.params
        for v in range(self.size):
            for h in range(self.size):
                vv = v + 1
                hh = h + 1

                wij = 1/(2*math.sqrt(2)*self.size*self.theta) * math.sqrt(vv*vv + hh*hh)
                wi0 = 1/(2*math.sqrt(2)*self.size*self.theta) * math.sqrt(0   + hh*hh)
                w0j = 1/(2*math.sqrt(2)*self.size*self.theta) * math.sqrt(vv*vv + 0)

                angle_pi = np.maximum(np.minimum(2*wi0*w0j/wij/wij, 1), -1)
                angle = np.abs(math.asin(angle_pi))
                A_f = a * (b+c*wij)*math.exp(-math.pow(c*wij, d))

                self.Tbase[v, h] = self.s/A_f*self.G/(self.Lmax - self.Lmin) \
                                    /self.s_pi_v[h]/self.s_pi_h[v]/(self.r+(1-self.r) \
                                    * math.pow(math.cos(angle), 2))
        return self.Tbase

class Enhancement():
    def __init__(self, block_size):
        self.block_size = block_size

        self.slicer = BlockSlicing(block_size, block_size)
        self.csf = CSF(size=block_size, Rvd=6, Pic=4, s=0.25, r=0.6, params=[2.6, 0.0192, 0.114, 1.1])
        self.Tbase = self.csf.makeTbase()
        self.Tbase2 = self.Tbase*self.Tbase

    def enhancing(self, in_img, strength, alpha):
        
        h, w = np.shape(in_img)
        h = np.int(np.floor(h/self.block_size) * self.block_size)
        w = np.int(np.floor(w/self.block_size) * self.block_size)
        img = np.array(in_img[:h,:w]).copy()

        blocks = self.slicer.slicing(img)
        n_slice = np.shape(blocks)[2]
        blocks_out = np.zeros([self.block_size, self.block_size, n_slice])
        ydct = np.zeros([self.block_size, self.block_size, n_slice])
        # ydct_abs = np.zeros([self.block_size, self.block_size, n_slice])
        # ydct_sq = np.zeros([self.block_size, self.block_size, n_slice])

        st_band = np.int(self.block_size/3)
        enhLambda = np.ones([self.block_size]) * strength
        enhLambda[:st_band] = 1

        for i in range(n_slice):
            ydct[:,:,i] = cv2.dct(blocks[:,:,i])

        ydct_abs = np.abs(ydct)
        ydct_sq = ydct * ydct

        Energy = np.sum(np.sum(ydct_abs, axis=1), axis=0) - ydct_abs[0,0,:] + 0.00001
        Hgrad = np.sum(np.sum(ydct_abs[1:, 0], axis=1), axis=0) / Energy + 0.00001
        Vgrad = np.sum(np.sum(ydct_abs[0, 1:], axis=1), axis=0) / Energy + 0.00001

        alpha_rooting = np.power( (ydct_abs+0.00001)/(ydct_abs[0, 0, :]+0.00001), alpha-1 )
        enh_sq = alpha_rooting * ydct * alpha_rooting * ydct

        ydct_sq_power = np.power(ydct_sq, 1/2.4)
        enh_sq_power = np.power(enh_sq, 1/2.4)

        Tbase_vsum = []
        Tbase_hsum = []
        for n in range(self.block_size-1):
            Tbase_vsum.append(np.sum(self.Tbase2[n+1,:]))
            Tbase_hsum.append(np.sum(self.Tbase2[:,n+1]))

        for i in range(n_slice):
            # ydct_abs = np.abs(ydct)
            # ydct_sq = ydct * ydct

            # Energy = np.sum(ydct_abs) - ydct_abs[0,0] + 0.00001
            # Hgrad = np.sum(ydct_abs[1:, 0]) / Energy + 0.00001
            # Vgrad = np.sum(ydct_abs[0, 1:]) / Energy + 0.00001

            # alpha_rooting = np.power( (ydct_abs+0.0001)/ydct_abs[0, 0], alpha-1 )
            # enh_sq = alpha_rooting * ydct * alpha_rooting * ydct

            oriEnergyVer = np.sum(ydct_sq[0,:,i]) - ydct_sq[0,0,i]
            enhEnergyVer = np.sum(ydct_sq[0,:,i]) - ydct_sq[0,0,i]

            oriEnergyHor = np.sum(ydct_sq[:,0,i]) - ydct_sq[0,0,i]
            enhEnergyHor = np.sum(ydct_sq[:,0,i]) - ydct_sq[0,0,i]

            enhDCT_ver = np.empty([self.block_size, self.block_size])
            enhDCT_ver[0,:] = ydct[0,:,i]

            enhDCT_hor = np.empty([self.block_size, self.block_size])
            enhDCT_hor[:,0] = ydct[:,0,i]

            for n in range(self.block_size-1):
                vv = n + 1
                # Tbase_vValue = np.sum(self.Tbase2[vv,:])
                # Rver = (Tbase_vsum[n] + enhEnergyVer + np.sum( np.power(enh_sq[vv, :, i], 2.4) ) ) \
                #        / (Tbase_vsum[n] + oriEnergyVer + np.sum( np.power(ydct_sq[vv, :, i], 2.4) ) )
                Rver = (Tbase_vsum[n] + enhEnergyVer + np.sum( enh_sq_power[vv, :, i] ) ) \
                       / (Tbase_vsum[n] + oriEnergyVer + np.sum( ydct_sq_power[vv, :, i] ) )
                enhDCT_ver[vv, :] = enhLambda * Rver * ydct[vv, :,i]
                enhEnergyVer += np.sum(enhDCT_ver[vv, :]*enhDCT_ver[vv, :])
                oriEnergyVer += np.sum(ydct_sq[vv, :,i])

                hh = n + 1
                # Tbase_hValue = np.sum(self.Tbase2[:,hh])
                RHor = (Tbase_hsum[n] + enhEnergyHor + np.sum( enh_sq_power[:, hh, i] ) ) \
                       / (Tbase_hsum[n] + oriEnergyHor + np.sum( ydct_sq_power[:, hh, i] ) )
                enhDCT_hor[:, hh] = enhLambda * RHor * ydct[:, hh,i]          
                enhEnergyHor += np.sum(enhDCT_hor[:,hh]*enhDCT_hor[:,hh])
                oriEnergyHor += np.sum(ydct_sq[:,hh,i])

            enhDCT = Hgrad[i] / (Hgrad[i]+Vgrad[i]) * enhDCT_hor + Vgrad[i] / (Hgrad[i]+Vgrad[i]) * enhDCT_ver
    
            # block_idct = cv2.idct(ydct[:,:,i])
            block_idct = cv2.idct(enhDCT)
            # blocks_out = np.concatenate((blocks_out, block_idct[..., np.newaxis]), axis=2)
            blocks_out[:,:,i] = block_idct
        # out_enh = self.slicer.unslicing(blocks, w, h)
        out_enh = self.slicer.unslicing(blocks_out, w, h)

        out_img = np.array(in_img).copy()
        out_img[:h,:w] = out_enh
        return out_img


block_size = 256
in_bgr = cv2.imread("Boat.png")
# in_bgr = cv2.resize(in_bgr,(block_size*4, (block_size)*3))

# h, w, c = np.shape(in_bgr)
# h = np.int(np.floor(h/block_size) * block_size)
# w = np.int(np.floor(w/block_size) * block_size)
# bgr = in_bgr[:h,:w,:]
# bgr = in_bgr

cv2.imshow('in_bgr', in_bgr)
cv2.waitKey(0)

yuv = bgr2yuv(in_bgr)
Y = np.array(yuv[:,:,0])

# slicer = BlockSlicing(block_size, block_size)
# csf = CSF(size=16, Rvd=6, Pic=4, s=0.25, r=0.6, params=[2.6, 0.0192, 0.114, 1.1])
# Tbase = csf.makeTbase()

enhancer = Enhancement(block_size = 16)
Y_out = enhancer.enhancing(Y, 1.5, 0.95)

# blocks = slicer.slicing(Y)
# n_slice = np.shape(blocks)[2]


# Ydct = np.empty([block_size, block_size, 0])
# blocks_out = np.empty([block_size, block_size, 0])

# st_band = np
# for i in range(n_slice):
#     ydct = cv2.dct[blocks[:,:,i]]
#     ydct_abs = np.abs(ydct)

#     Energy = np.sum(ydct_abs) - ydct_abs[0,0] + 0.00001
#     Hgrad = np.sum(ydct_abs[1:,0]) / Energy
#     Vgrad = np.sum(ydct_abs[0, 1:]) / Energy

#     R = np.empty([block_size, block_size])


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

# for i in range(n_slice):
#     cv2.imshow("block", blocks[:,:,i]/255)
#     cv2.waitKey(0)



# Y_out = slicer.unslicing(blocks, w, h)



# blocks_out = np.reshape(blocks, (block_size, h,-1))
# Y_out      = np.reshape(np.ravel(blocks_out, order='F'), (h, w), order='F')

# Y_out = blocks_out.reshape(h,w)

cv2.imshow("Y", Y/255)
cv2.imshow("Y_out", Y_out/255)
cv2.waitKey(0)

# yuv[:,:,0] = np.uint8(Y_out)
# bgr2 = yuv2bgr(yuv)
# out_bgr = np.asanyarray(in_bgr)
# out_bgr[:h,:w,:] = bgr2
# cv2.imwrite("Enh_out.png", out_bgr)