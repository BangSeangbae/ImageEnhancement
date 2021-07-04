import numpy as np
from numpy.core.shape_base import block
import cv2
from functions import *
import argparse

parser = argparse.ArgumentParser(description="Process Arguments")
parser.add_argument('--img', required=False, type=str, default='./test.png', help='Location of Input Image')
parser.add_argument('--l', required=False, type=float, default=1.8, help='Enhancement Strength Lambda')
parser.add_argument('--a', required=False, type=float, default=0.88, help='Alpha Rooting Value')
parser.add_argument('--block_size', required=False, type=int, default=16, help='DCT block size')
args = parser.parse_args()

class Enhancement():
    def __init__(self, block_size):
        self.block_size = block_size
        self.slicer = BlockSlicing(block_size, block_size)
        self.csf = CSF(size=block_size, Rvd=6, Pic=4, s=0.25, r=0.6, params=[2.6, 0.0192, 0.114, 1.1])
        self.Tbase = self.csf.makeTbase()
        self.Tbase2 = self.Tbase*self.Tbase
        self.R_power = 1/2.4

    def enhancing(self, in_img, strength, alpha):
        
        h, w = np.shape(in_img)
        h = np.int(np.floor(h/self.block_size) * self.block_size)
        w = np.int(np.floor(w/self.block_size) * self.block_size)
        img = np.array(in_img[:h,:w]).copy()

        blocks = self.slicer.slicing(img)
        n_slice = np.shape(blocks)[2]
        blocks_out = np.zeros([self.block_size, self.block_size, n_slice])
        ydct = np.zeros([self.block_size, self.block_size, n_slice])

        st_band = np.int(self.block_size/3)+1
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
        enh_sq = alpha_rooting * alpha_rooting * ydct_sq

        # ydct_sq_power = np.power(ydct_sq, 1/2.4)
        # enh_sq_power = np.power(enh_sq, 1/2.4)

        Tbase_vsum = []
        Tbase_hsum = []
        for n in range(self.block_size-1):
            Tbase_vsum.append(np.sum(self.Tbase2[n+1,:]))
            Tbase_hsum.append(np.sum(self.Tbase2[:,n+1]))

        for i in range(n_slice):
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
                Rver = (Tbase_vsum[n] + oriEnergyVer + np.sum( enh_sq[vv, :, i] ) ) \
                       / (Tbase_vsum[n] + oriEnergyVer + np.sum( ydct_sq[vv, :, i] ) )
                Rver = np.power(Rver, self.R_power)
                enhDCT_ver[vv, :] =  enhLambda * Rver * ydct[vv, :, i]
                enhEnergyVer += np.sum(enhDCT_ver[vv, :]*enhDCT_ver[vv, :])
                oriEnergyVer += np.sum(ydct_sq[vv, :,i])

                hh = n + 1
                Rhor = (Tbase_hsum[n] + enhEnergyHor + np.sum( enh_sq[:, hh, i] ) ) \
                       / (Tbase_hsum[n] + oriEnergyHor + np.sum( ydct_sq[:, hh, i] ) )
                Rhor = np.power(Rhor, self.R_power)
                enhDCT_hor[:, hh] = enhLambda * Rhor * ydct[:, hh, i]          
                enhEnergyHor += np.sum(enhDCT_hor[:,hh]*enhDCT_hor[:,hh])
                oriEnergyHor += np.sum(ydct_sq[:,hh,i])

            enhDCT = Hgrad[i] / (Hgrad[i]+Vgrad[i]) * enhDCT_hor + Vgrad[i] / (Hgrad[i]+Vgrad[i]) * enhDCT_ver
            # enhDCT = enhDCT_ver
    
            block_idct = cv2.idct(enhDCT)
            blocks_out[:,:,i] = block_idct
        out_enh = self.slicer.unslicing(blocks_out, w, h)

        out_img = np.array(in_img).copy()
        out_img[:h,:w] = out_enh
        return out_img

if __name__ == "__main__":
    in_bgr = cv2.imread("Boat.png")

    in_bgr = cv2.resize(in_bgr,(1280, 720))

    h, w, c = np.shape(in_bgr)

    # cv2.imshow('in_bgr', in_bgr)
    # cv2.waitKey(0)

    yuv = bgr2yuv(in_bgr)
    Y = np.array(yuv[:,:,0])

    enhancer = Enhancement(block_size = args.block_size)
    Y_out = enhancer.enhancing(Y, args.l, args.a)

    cv2.imshow("Y", Y/255)
    cv2.imshow("Y_out", Y_out/255)
    cv2.waitKey(0)

    yuv[:,:,0] = np.uint8(Y_out)
    bgr2 = yuv2bgr(yuv)
    out_bgr = np.asanyarray(in_bgr)
    out_bgr[:h,:w,:] = bgr2

    cv2.imwrite("Enh_out.png", out_bgr)