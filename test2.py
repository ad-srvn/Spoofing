import os
import librosa
import numpy as np
import spafe.features.lpc
from sklearn.mixture import GaussianMixture

import numpy as np
import librosa
import scipy.signal

def compute_lpcc(audio_signal, sample_rate, num_coefficients=13):
    # Step 1: Pre-emphasis
    pre_emphasized_signal = scipy.signal.lfilter([1, -0.97], 1, audio_signal)
    
    # Step 2: Frame the signal
    frame_size = 0.025  # Frame size in seconds (adjust as needed)
    frame_stride = 0.01  # Frame stride in seconds (adjust as needed)
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    
    frames = librosa.util.frame(pre_emphasized_signal, frame_length=frame_length, hop_length=frame_step)
    
    lpcc_coefficients = []
    
    for frame in frames.T:
        # Step 3: Windowing (e.g., Hamming window)
        windowed_frame = frame * np.hamming(len(frame))
        
        # Step 4: Autocorrelation
        autocorr = np.correlate(windowed_frame, windowed_frame, mode='full')
        
        # Step 5: Levinson-Durbin Algorithm
        lp_coefficients = librosa.core.lpc(autocorr, order=num_coefficients - 1)
        
        # Step 6: Cepstral Analysis (LPCCs)
        lpccs = np.fft.ifft(np.log(np.abs(np.fft.fft(lp_coefficients, n=num_coefficients))))
        lpcc_coefficients.append(lpccs.real)
    
    return np.array(lpcc_coefficients)

def extract_features(file_path, sr):
    # Extract multiple features for a given audio file
    speech, _ = librosa.load(file_path, sr=sr)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13)
    # num_ceps=13,, win_type='hann' win_len=0.025, win_hop=0.01,, dither=1
    # ,pre_emph=1, pre_emph_coeff=0.97,lifter=1, normalize=1)
    # lpcc = spafe.features.lpc.lpcc(speech, fs=sr)
    lpcc = compute_lpcc(speech,sr,num_coefficients=13)
    # print(len(np.vstack(lpcc)[0]))
    # Transpose to have shape (time_steps, n_mfcc)
    combined_features = np.vstack((mfcc.T,lpcc))  # You can add more features if needed
    # print(len(combined_features[0]))
    return combined_features

# Function to train the speaker recognition system
def speaker_recognition_system(dataset_path):

    sr=16000
    speakers = []
    feature_sets = []
    num_gmm_components = 4  # You can adjust the number of Gaussian components
    
    # Step 1: Load the dataset (Assuming the dataset is organized in folders, one folder per speaker)
    for speaker_folder in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_folder)

        if os.path.isdir(speaker_path):
            speakers.append(speaker_folder)
            features = []
            # Step 2: Extract MFCC features for each speech sample
            for file_name in os.listdir(speaker_path):  
                file_path = os.path.join(speaker_path, file_name)
                # speech, _ = librosa.load(file_path, sr=sr)  # sr=None means no resampling, use the original sampling rate
                # mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13)  # n_mfcc is the number of MFCC coefficients
                # features.append(mfcc.T)  
                features.append(extract_features(file_path, sr))
            # mfcc_features.append(np.vstack(features))
            # for i in features:
            #     print(len(i))
            combined_features = np.vstack(features)
            # print(len(combined_features))
            feature_sets.append(combined_features)
    # Step 3: Train a GMM for each speaker
    gmm_models = []
    for features in feature_sets:
        gmm = GaussianMixture(n_components=num_gmm_components, covariance_type='diag')
        gmm.fit(features)
        gmm_models.append(gmm)

    # Save the trained models for future use (optional)
    # You can use the joblib library to save and load the models
    # import joblib
    # joblib.dump((gmm_models, speakers), 'speaker_models.pkl')

    print("Speaker recognition system trained successfully.")
    return gmm_models, speakers

# Function to identify the speaker of a given speech sample
def identify_speaker(file_path, gmm_models, speakers):
    
    sr=16000
    # Step 4: Extract MFCC features from the speech sample
    speech_sample, _ = librosa.load(file_path, sr=sr)
    mfcc_sample = librosa.feature.mfcc(y=speech_sample, sr=sr, n_mfcc=13).T

    # Step 5: Compute the likelihood of the sample belonging to each speaker using the GMMs
    likelihoods = [gmm.score(mfcc_sample) for gmm in gmm_models]

    # Step 6: Identify the speaker with the highest likelihood
    max_likelihood_idx = np.argmax(likelihoods)
    identified_speaker = speakers[max_likelihood_idx]

    return likelihoods

#MAIN EXECUTES
gmm_models, speakers = speaker_recognition_system('data')
# print(len(mfcc[0]))
# print(gmm_models,speakers)
print(identify_speaker('data\Speaker_1\A1.wav',gmm_models,speakers))
