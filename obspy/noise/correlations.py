import numpy as np
from scipy.signal import hilbert
#==================================================================================================
# Phase cross correlation (Schimmel 1999)
#==================================================================================================    

import time

def phase_xcorrelation(dat1, dat2, max_lag=10, nu=1):
    
    # Initialize arrays
    s1=np.zeros((len(dat1),),  dtype=np.float)
    s2=np.zeros((len(dat1),),  dtype=np.float)
   
    #Max lag in sec --> convert to sample numbers
    Fs=dat1.stats.sampling_rate
    max_lag=int(max_lag*Fs)
   
    return phase_xcorr(dat1.data,dat2.data,max_lag,nu)
   
             

#==================================================================================================
# Phase cross correlation (Schimmel 1999); this is obtained with a variable window length
#=================================================================================================  

def phase_xcorr(data1,data2,max_lag,nu=1):
    """
    
    data1, data2: Numpy arrays containing the analytic signal normalized sample by sample
    by their absolute value (ie containing only the instantaneous phase information)
    max_lag: maximum lag in number of samples, integer
    
    """
    
    #Initialize pcc array:
    pxc=np.zeros((2*max_lag+1,), dtype=float)
    
    
    
    #Obtain analytic signal
    data1=hilbert(data1)
    data2=hilbert(data2)
    
    
    #Normalization
    data1=data1/(np.abs(data1))
    data2=data2/(np.abs(data2))
    
   
    for k in range(0,max_lag+1):
        i11=0
        i12=len(data1)-k
        i21=k
        i22=len(data1)
        
        
        pxc[max_lag+k]=1.0/float(2*len(data1)-k)*(np.sum(np.abs(data1[i11:i12]+data2[i21:i22])**nu) - np.sum(np.abs(data1[i11:i12]-data2[i21:i22])**nu))
        pxc[max_lag-k]=1.0/float(2*len(data1)-k)*(np.sum(np.abs(data1[i21:i22]+data2[i11:i12])**nu) - np.sum(np.abs(data1[i21:i22]-data2[i11:i12])**nu))

    return pxc

