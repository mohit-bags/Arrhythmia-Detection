import neurokit2 as nk
import os

def detect_r_peaks():
    label_ecg = '/arrhythmia/ECGData/'
    denoised_files = os.listdir(label_ecg)

    for i in denoised_files:
        data = label_ecg+i
        df = pd.read_csv(data,header=None)
        df.columns=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6', 'label']
        
        cols = df.columns
        df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric, errors='coerce') 

      
        # Extract R-peaks locations
        _, rpeaks = nk.ecg_peaks(df["aVL"][1:], sampling_rate=500)

        # Visualize R-peaks in ECG signal
        plot = (nk.events_plot(rpeaks['ECG_R_Peaks'], df["aVL"]))
       
    return