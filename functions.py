import numpy as np
import math

def yuv2rgb(yuv):
    # HDTV with BT.709
    y = yuv[:,:,0]
    u = yuv[:,:,1]
    v = yuv[:,:,2]

    r = y               + 1.28033 * v
    g = y - 0.21482 * u - 0.38059 * v
    b = y + 2.12798 * u

    rgb = np.stack((r,g,b), axis=2)
    return rgb

def yuv2bgr(yuv):
    # HDTV with BT.709
    y = yuv[:,:,0]
    u = yuv[:,:,1]
    v = yuv[:,:,2]

    r = y               + 1.28033 * v
    g = y - 0.21482 * u - 0.38059 * v
    b = y + 2.12798 * u

    bgr = np.stack((b,g,r), axis=2)
    return bgr

def rgb2yuv(rgb): 
    # HDTV with BT.709
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    y =  0.2126  * r + 0.7152  * g + 0.0722  * b
    u = -0.09991 * r - 0.33609 * g + 0.436   * b
    v =  0.615   * r - 0.55861 * g - 0.05639 * b

    yuv = np.stack((y,u,v), axis=2)
    return yuv

def bgr2yuv(rgb):
    # HDTV with BT.709
    r = rgb[:,:,2]
    g = rgb[:,:,1]
    b = rgb[:,:,0]

    y =  0.2126  * r + 0.7152  * g + 0.0722  * b
    u = -0.09991 * r - 0.33609 * g + 0.436   * b
    v =  0.615   * r - 0.55861 * g - 0.05639 * b

    yuv = np.stack((y,u,v), axis=2)
    return yuv




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
