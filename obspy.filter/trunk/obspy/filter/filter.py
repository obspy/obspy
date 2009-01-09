from scipy.signal import iirfilter,lfilter


def bandpass(data,freqmin,freqmax,df=200,corners=4):
  """Butterworth-Bandpass Filter
  
  Filter data from freqmin to freqmax using
  corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, [freqmin/fe, freqmax/fe], btype='band',ftype='butter',output='ba')
  return lfilter(b, a, data)

def bandpassZPHSH(data,freqmin,freqmax,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Bandpass Filter
  
  Filter data from freqmin to freqmax using corners corners and doing 2 runs
  over the data, one from left to right and one from right to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=bandpass(data, freqmin, freqmax, df, corners)
  x=bandpass(x[::-1], freqmin, freqmax, df, corners)
  return x[::-1]

def bandstop(data,freqmin,freqmax,df=200,corners=4):
  """Butterworth-Bandstop Filter
  
  Filter data removing data between frequencies freqmin and freqmax using
  corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, [freqmin/fe, freqmax/fe], btype='bandstop',ftype='butter',output='ba')
  return lfilter(b, a, data)

def bandstopZPHSH(data,freqmin,freqmax,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Bandstop Filter
  
  Filter data removing data between frequencies freqmin and freqmax using
  corners corners and doing 2 runs over the data, one from left to right and
  one from right to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=bandstop(data, freqmin, freqmax, df, corners)
  x=bandstop(x[::-1], freqmin, freqmax, df, corners)
  return x[::-1]

def lowpass(data,freq,df=200,corners=4):
  """Butterworth-Lowpass Filter
  
  Filter data removing data over certain frequency freq using corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, freq/fe, btype='lowpass',ftype='butter',output='ba')
  return lfilter(b, a, data)

def lowpassZPHSH(data,freq,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Lowpass Filter
  
  Filter data removing data over certain frequency freq using corners corners
  and doing 2 runs over the data, one from left to right and one from right
  to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=lowpass(data, freq, df, corners)
  x=lowpass(x[::-1], freq, df, corners)
  return x[::-1]

def highpass(data,freq,df=200,corners=4):
  """Butterworth-Highpass Filter:
  
  Filter data removing data below certain frequency freq using corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, freq/fe, btype='highpass',ftype='butter',output='ba')
  return lfilter(b, a, data)

def highpassZPHSH(data,freq,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Highpass Filter:
  
  Filter data removing data below certain frequency freq using corners corners
  and doing 2 runs over the data, one from left to right and one from right
  to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=highpass(data, freq, df, corners)
  x=highpass(x[::-1], freq, df, corners)
  return x[::-1]
