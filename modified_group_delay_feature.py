#function [grp_phase, cep, ts] = 
import scipy as sp
import librosa
import numpy as np
import math
import numpy.matlib as npm
from median_filter import median_filter
import warnings
warnings.filterwarnings("ignore")
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def modified_group_delay_feature(file_name, rho=0.4, gamma=0.9, num_coeff=12, frame_shift=0.01):
    #input: 
    #     file_name: path for the waveform. The waveform should have a header
    #     rho: a parameter to control the shape of modified group delay spectra
    #     gamma: a parameter to control the shape of the modified group delay spectra
    #     num_coeff: the desired feature dimension
    #     [frame_shift]: 
    #
    #output:
    #     grp_phase: modifed gropu delay spectrogram
    #     cep: modified group delay cepstral feature.
    #     ts: time instants at the center of each analysis frame.
    #
    #Example:
    #     [grp_phase, cep, ts] = modified_group_delay_feature('./100001.wav', 0.4, 0.9, 12);
    # Please tune rho and gamma for better performance
    #     See also: howtos/HOWTO_features.m   
    #
    # by Zhizheng Wu (zhizheng.wu@ed.ac.uk)
    # http://www.zhizheng.org
    #
    # The code has been used in the following three papers:
    # Zhizheng Wu, Xiong Xiao, Eng Siong Chng, Haizhou Li, "Synthetic speech detection using temporal modulation feature", IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2013.
    # Zhizheng Wu, Tomi Kinnunen, Eng Siong Chng, Haizhou Li, Eliathamby Ambikairajah, "A study on spoofing attack in state-of-the-art speaker verification: the telephone speech case", Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) 2012. 
    # Zhizheng Wu, Eng Siong Chng, Haizhou Li, "Detecting Converted Speech and Natural Speech for anti-Spoofing Attack in Speaker Recognition", Interspeech 2012. 
    #
    # feel free to modify the code and welcome to cite above papers :)

    [speech,fs]  = librosa.load(file_name,sr=None);

    frame_length = 0.025; #msec
    
    NFFT         = 512;
    pre_emph     = True;

    ### Pre-emphasis + framing 
    if pre_emph==True:
        speech = librosa.effects.preemphasis(speech);
    
    frame_length = round((frame_length)*fs);

    frame_shift = round((frame_shift)*fs);

    frames = librosa.util.frame(x=speech, frame_length=frame_length, hop_length=frame_shift);

    #ts = (ts-1)/fs;

    frame_num    = len(frames);
    frame_length = len(frames[0]);
    delay_vector = range(0,frame_length);
    delay_matrix = npm.repmat(delay_vector, frame_num, 1);

    delay_frames = np.multiply(frames,delay_matrix);

    x_spec = sp.fft.fft(frames.T, NFFT);
    y_spec = sp.fft.fft(delay_frames.T, NFFT);
    x_spec = x_spec[0:int(NFFT/2)+1, :];
    y_spec = y_spec[0:int(NFFT/2)+1, :];

    l=0
    for i in x_spec:
        for j in i:
            l+=1
    temp_x_spec = abs(x_spec);
    b=np.multiply(np.real(x_spec),np.real(y_spec))
    c=np.multiply(np.imag(y_spec),np.imag(x_spec))
    a=np.add(b,c)
    me=median_filter(np.log10(abs(x_spec)),5);
    dct_spec=sp.fft.dct(x=me);
    smooth_spec = sp.fft.idct(x=dct_spec);
    grp_phase1=np.divide(a,np.power(np.power(smooth_spec,2.18281828),2*rho))
    grp_phase=np.multiply(np.divide(grp_phase1,np.abs(grp_phase1)),np.power(np.abs(grp_phase1),gamma))
    grp_phase=np.divide(grp_phase,np.max(np.max(np.abs(grp_phase))))
    grp_phase=np.nan_to_num(grp_phase)
    cep=sp.fft.dct(grp_phase)
    cep = cep[1:num_coeff+1, :].T;
    return [grp_phase, cep]
