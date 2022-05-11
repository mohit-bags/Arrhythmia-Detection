import matplotlib.pyplot as plt
import numpy as np
import pywt
#De-trending

rr = [0.155,0.16,0.243,0.258,0.278,0.156,0.172,0.246,0.257,0.26,0.193,0.329,0.282,0.27,0.173,0.151,0.164,0.204,0.3,0.18,0.262]

def detrend(rr, wavelet, level):

    coefficients = pywt.wavedec(rr, wavelet)
    tau = np.std(coefficients[-level])*np.sqrt(2*np.log(len(rr)))
    coefficients[1:] = (pywt.threshold(i, value=tau, mode="soft") for i in coefficients[1:])
    y = pywt.waverec( coefficients, wavelet)

    return y, rr-y
print("Ori",rr,"\n\n")
y, x = detrend(rr, "db25", 1)
print("rr-y:",x)
print("\ny\n",y)





