#!/usr/bin/env python3
import numpy as np
import tempfile
import copy
import librosa
import librosa as rs
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
import soundfile as sf
import scipy
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample, lfilter

# vocal tract length normalization
def vtln(x, coef = 0.):
  # STFT
  mag, phase = rs.magphase(rs.core.stft(x))
  mag, phase = np.log(mag).T, phase.T

  # Frequency
  freq = np.linspace(0, np.pi, mag.shape[1]) 
  freq_warped = freq + 2.0 * np.arctan(coef * np.sin(freq) / (1 - coef * np.cos(freq)))
  
  # Warping
  mag_warped = np.zeros(mag.shape, dtype = mag.dtype)
  for t in range(mag.shape[0]):
    mag_warped[t, :] = np.interp(freq, freq_warped, mag[t, :])

  # ISTFT
  y = np.real(rs.core.istft(np.exp(mag_warped).T * phase.T)).astype(x.dtype)

  return y


# Mcadams transformation: Baseline2 of VoicePrivacy2020
def vp_baseline2(x, mcadams = 0.8, winlen = int(20 * 0.001 * 16000), shift = int(10 * 0.001 * 16000), lp_order = 20):
  eps = np.finfo(np.float32).eps
  x2 = copy.deepcopy(x) + eps
  length_x = len(x2)
  
  # FFT parameters
  # n_fft = 2**(np.ceil((np.log2(winlen)))).astype(int)
  wPR = np.hanning(winlen)
  K = np.sum(wPR)/shift
  win = np.sqrt(wPR/K)
  n_frame = 1+np.floor((length_x-winlen)/shift).astype(int) # nr of complete frames
  
  # carry out the overlap - add FFT processing
  y = np.zeros([length_x])

  for m in np.arange(1, n_frame):
    # indices of the mth frame
    index = np.arange(m*shift,np.minimum(m*shift+winlen,length_x))    
    # windowed mth frame (other than rectangular window)
    frame = x2[index]*win 
    # get lpc coefficients
    a_lpc = rs.lpc(frame+eps,lp_order)
    # get poles
    poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
    #index of imaginary poles
    ind_imag = np.where(np.isreal(poles)==False)[0]
    #index of first imaginary poles
    ind_imag_con = ind_imag[np.arange(0,np.size(ind_imag),2)]
    
    # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
    # values >1 expand the spectrum, while values <1 constract it for angles>1
    # values >1 constract the spectrum, while values <1 expand it for angles<1
    # the choice of this value is strongly linked to the number of lpc coefficients
    # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
    # a smaller lpc coefficients number allows for a bigger flexibility
    new_angles = np.angle(poles[ind_imag_con])**mcadams
    
    # make sure new angles stay between 0 and pi
    new_angles[np.where(new_angles>=np.pi)] = np.pi        
    new_angles[np.where(new_angles<=0)] = 0  
    
    # copy of the original poles to be adjusted with the new angles
    new_poles = poles
    for k in np.arange(np.size(ind_imag_con)):
      # compute new poles with the same magnitued and new angles
      new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]])*np.exp(1j*new_angles[k])
      # applied also to the conjugate pole
      new_poles[ind_imag_con[k]+1] = np.abs(poles[ind_imag_con[k]+1])*np.exp(-1j*new_angles[k])
        
    # recover new, modified lpc coefficients
    a_lpc_new = np.real(np.poly(new_poles))
    # get residual excitation for reconstruction
    res = lfilter(a_lpc,np.array(1),frame)
    # reconstruct frames with new lpc coefficient
    frame_rec = lfilter(np.array([1]),a_lpc_new,res)
    frame_rec = frame_rec*win    
    
    outindex = np.arange(m*shift,m*shift+len(frame_rec))
    # overlap add
    y[outindex] = y[outindex] + frame_rec
      
  y = y/np.max(np.abs(y))
  return y.astype(x.dtype)

def _trajectory_smoothing(x, thresh = 0.5):
  y = copy.copy(x)

  b, a = signal.butter(2, thresh)
  for d in range(y.shape[1]):
    y[:, d] = signal.filtfilt(b, a, y[:, d])
    y[:, d] = signal.filtfilt(b, a, y[::-1, d])[::-1]

  return y

# modulation spectrum smoothing
def modspec_smoothing(x, coef = 0.1):
  # STFT
  mag_x, phase_x = rs.magphase(rs.core.stft(x))
  mag_x, phase_x = np.log(mag_x).T, phase_x.T
  mag_x_smoothed = _trajectory_smoothing(mag_x, coef)

  # ISTFT
  y = np.real(rs.core.istft(np.exp(mag_x_smoothed).T * phase_x.T)).astype(x.dtype)
  y = y * np.sqrt(np.sum(x * x)) / np.sqrt(np.sum(y * y))
  
  return y

