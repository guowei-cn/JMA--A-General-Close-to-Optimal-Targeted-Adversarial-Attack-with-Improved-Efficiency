import numpy as np
import math

def compute_psnr(img1,img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if img1.max() > 1 and img1.max() % 1 == 0:
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        return 20 * math.log10(1 / math.sqrt(mse))

def RMSD2PSNR(rmsd,PIXEL_MAX=1):
    #rmsd=rootedmeansquaredistance
    #rmsd=sqrt(mean((a-b)**2))=sqrt(mse)
    mse = rmsd**2
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))
