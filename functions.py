import numpy as np


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
