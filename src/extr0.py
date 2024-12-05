import wfdb
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
import biosppy.signals.ecg as ecg

# Load the dataset using WFDB
record_path = "C:/Users/user/Desktop/Liwia/5term/Enhance/00001_lr"
record = wfdb.rdrecord(record_path)
ecg_signal = record.p_signal  # ECG signals (12 leads, array)
fs = record.fs  # Sampling frequency

# Extract lead names
lead_names = record.sig_name

    # Heart rate (RR interval):

def heart_rate(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    rr_intervals = np.diff(out['rpeaks']) / fs  # RR intervals in seconds
    heart_rate = 60 / rr_intervals  # HR in beats per minute (bpm)
    avg_heart_rate = np.mean(heart_rate)

    return avg_heart_rate

    # QT interval (duration of QRST):

def QT_duration(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    q_position = ecg.getQPositions(ecg_proc=out, show=False)
    q_onsets = q_position['Q_start_positions']
    t_position = ecg.getTPositions(ecg_proc=out, show=False)
    t_offsets = t_position['T_end_positions']

    qt_intervals = []
    for q_onset, t_offset in zip(q_onsets, t_offsets):
        qt_interval = (t_offset - q_onset) / fs
        qt_intervals.append(qt_interval)

    avg_qt_interval = np.mean(qt_intervals)*1000

    return avg_qt_interval

    # ST segment (el/depr):

def ST_amplitude(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    s_position = ecg.getSPositions(ecg_proc=out, show=False)
    s_offsets = s_position['S_end_positions']
    t_position = ecg.getTPositions(ecg_proc=out, show=False)
    t_onsets = t_position['T_start_positions']

    if len(t_onsets) != len(s_offsets):
        min_len = min(len(t_onsets), len(s_offsets))
        t_onsets = t_onsets[:min_len]
        s_offsets = s_offsets[:min_len]

    st_amplitudes = []
    for s_offset, t_onset in zip(s_offsets, t_onsets):
        s_value = signal[s_offset]  # S-wave amplitude (end of S-wave)
        t_value = signal[t_onset]  # T-wave amplitude (start of T-wave)

        st_amplitude = t_value - s_value
        st_amplitudes.append(st_amplitude)

    avg_st_amplitude = np.mean(st_amplitudes)

    return avg_st_amplitude

    # PR interval (duration):

def PR_duration(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    p_wave_positions = ecg.getPPositions(ecg_proc=out, show=False)
    p_onsets = p_wave_positions['P_start_positions']

    r_peaks = out['rpeaks']

    qrs_onsets = []
    for r_peak in r_peaks:
        # Find the local minimum before the R-peak (to approximate the QRS onset)
        start_index = max(0, r_peak - int(0.15 * fs))  # 150 ms before R-peak
        end_index = r_peak
        qrs_onset = np.argmin(signal[start_index:end_index]) + start_index
        qrs_onsets.append(qrs_onset)

    pr_intervals = []
    for p_onset in p_onsets:
        # Find the nearest QRS onset after the P-wave onset
        closest_qrs_onset = min(qrs_onsets, key=lambda q: abs(q - p_onset))
        pr_interval = (closest_qrs_onset - p_onset) / fs  # Convert to seconds
        pr_intervals.append(pr_interval)

    avg_pr_interval = np.mean(pr_intervals)*1000

    return avg_pr_interval

    # P-wave duration: (II, V1, aVF)

def P_wave(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    p_wave_positions = ecg.getPPositions(ecg_proc=out, show=False)
    p_onsets = p_wave_positions['P_start_positions']
    p_offsets = p_wave_positions['P_end_positions']

    if len(p_onsets) != len(p_offsets):
        min_len = min(len(p_onsets), len(p_offsets))
        p_onsets = p_onsets[:min_len]
        p_offsets = p_offsets[:min_len]

    p_wave_durations_samples = np.array(p_offsets) - np.array(p_onsets)
    avg_p_wave_durations = np.mean(p_wave_durations_samples * 1000 / fs)

    return avg_p_wave_durations

    # R-wave amplitude:

def R_wave_amplitude(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    r_peak_value = np.max(signal[out['rpeaks']])  # Maximum value at the R-peaks
    baseline = np.mean(signal[:1000])  # Example: baseline as the mean of the first 1000 samples (pre-R-peak)

    r_amplitude = r_peak_value - baseline
    mean_r_amplitude = np.mean(r_amplitude)

    return mean_r_amplitude

    # R-wave duration: (maybe not???)

def R_wave_duration(ecg_signal, fs, lead_index):
    signal = ecg_signal[:, lead_index]
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    r_peaks = out['rpeaks']
    window_size = int(0.03 * fs)

    r_onsets = []
    r_offsets = []

    for r_peak in r_peaks:
        start_idx = max(0, r_peak - window_size)
        end_idx = min(len(signal), r_peak + window_size)

        r_onset = start_idx + np.argmin(signal[start_idx:r_peak])
        r_onsets.append(r_onset)

        r_offset = r_peak + np.argmin(signal[r_peak:end_idx])
        r_offsets.append(r_offset)

    if len(r_onsets) != len(r_offsets):
        min_len = min(len(r_onsets), len(r_offsets))
        r_onsets = r_onsets[:min_len]
        r_offsets = r_offsets[:min_len]

    r_wave_durations_samples = np.array(r_offsets) - np.array(r_onsets)
    avg_r_wave_duration = np.mean(r_wave_durations_samples * 1000 / fs)

    return avg_r_wave_duration

    # Q-wave duration: (I, aVL)


    # T-wave amplitude/shape:





# method to print from all leads !!


lead_index = 0
hr = heart_rate(ecg_signal, fs, lead_index)
p_wave_duration = P_wave(ecg_signal, fs, lead_index)
r_amplitude = R_wave_amplitude(ecg_signal, fs, lead_index)
st_amplitude = ST_amplitude(ecg_signal, fs, lead_index)
pr_duration = PR_duration(ecg_signal, fs, lead_index)
qt_duration = QT_duration(ecg_signal, fs, lead_index)


print(f"Lead {lead_index} - Heart Rate: {hr:.2f} bpm")
print(f"Lead {lead_index} - P-wave Duration: {p_wave_duration:.2f} ms")
print(f"Lead {lead_index} - R-wave Amplitude: {r_amplitude:.2f} mV")
print(f"Lead {lead_index} - ST-wave Amplitude: {st_amplitude:.2f} mV")
print(f"Lead {lead_index} - PR-wave Duration: {pr_duration:.2f} ms")
print(f"Lead {lead_index} - QT-wave Duration: {qt_duration:.2f} ms")
