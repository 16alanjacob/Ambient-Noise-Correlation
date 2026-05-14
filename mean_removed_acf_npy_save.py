from scipy import fftpack
import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert
from obspy import read
import glob
import os

root_path = r"D:\Alan CUSAT- dissertation\TRANSCORBE\9H"
output_root = r"D:\Alan CUSAT- dissertation\TRANSCORBE\ACF_daily_corrected"
os.makedirs(output_root, exist_ok=True)

def comb_filter(trace, fs):
    p1 = [0.5, 2, 6]
    p2 = [2, 6, 20]
    nyq = fs / 2
    weights = []
    for T1, T2 in zip(p1, p2):
        fmax = 1.0 / T1
        fmin = 1.0 / T2
        if fmax >= nyq:
            continue
        b, a = signal.butter(2, [fmin, fmax], btype="band", fs=fs)
        tr = signal.filtfilt(b, a, trace)
        env = np.abs(hilbert(tr))
        weights.append(1.0 / (env + 1e-10))
    if not weights:
        return trace / (np.max(np.abs(trace)) + 1e-10)
    W = np.ones_like(trace)
    for w in weights:
        W *= w
    water_level = 0.10 * np.mean(W)
    W[W < water_level] = water_level
    trace_norm = trace * W
    return trace_norm / (np.max(np.abs(trace_norm)) + 1e-10)

def autocorrelation(trace, maxlag):
    trace = trace - np.mean(trace)
    lentrace = len(trace)
    goodnumber = int(maxlag + lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[:lentrace] = trace
    tr2[:lentrace] /= np.sqrt(np.sum(tr2[:lentrace] ** 2))
    tr2 = fftpack.fft(tr2)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:maxlag + lentrace] = trace
    tr1[maxlag:maxlag + lentrace] /= np.sqrt(np.sum(tr1[maxlag:maxlag + lentrace] ** 2))
    tr2 *= fftpack.fft(tr1)
    tr2[np.isnan(tr2)] = 0
    tr2[np.isinf(tr2)] = 0
    acf = fftpack.ifft(tr2).real
    return acf[:2 * maxlag + 1].astype(np.float32)

def process_station(station_path):
    files = sorted(glob.glob(os.path.join(station_path, "*.mseed")))
    if len(files) == 0:
        return
    daily_acfs = []
    station_name = None
    maxlag_sec = 20
    for file in files:
        try:
            st = read(file)
            tr = st[0]
            if len(tr.data) < 200:
                print("  skipped short trace:", file)
                return
            station_name = tr.stats.station
            TARGET_DT = 0.2
            if abs(tr.stats.delta - TARGET_DT) > 1e-5:
                tr.resample(1 / TARGET_DT)
            fs = tr.stats.sampling_rate
            tr.detrend("linear")
            tr.detrend("demean")
            tr.filter("bandpass",
                      freqmin=0.1,
                      freqmax=2,
                      zerophase=True)
            tr.taper(0.05)
            raw = tr.data.astype(np.float32)
            norm = comb_filter(raw, fs)
            maxlag = int(maxlag_sec * fs)
            acf = autocorrelation(norm, maxlag)
            TARGET_LEN = int(2 * maxlag_sec / TARGET_DT) + 1
            acf = acf[:TARGET_LEN]
            daily_acfs.append(acf)
            print("  done:", os.path.basename(file))
        except Exception as e:
            print("  error:", file)
            print(e)
    if len(daily_acfs) == 0:
        return
    print(f"\nProcessing station: {station_name} ({len(files)} files)")
    daily_acfs = np.vstack(daily_acfs).astype(np.float32)
    mean_acf = np.mean(daily_acfs, axis=0)
    station_outdir = os.path.join(output_root, station_name)
    os.makedirs(station_outdir, exist_ok=True)
    for i, acf in enumerate(daily_acfs):
        acf_residual = acf - mean_acf
        acf_residual = acf_residual.astype(np.float32)
        outfile = os.path.join(
            station_outdir,
            f"{station_name}_day{i+1:04d}_acf.npy")
        if os.path.exists(outfile):
            print("  exists, skipping:", outfile)
            continue
        np.save(outfile, acf_residual)
        print("  saved:", outfile)
station_dirs = sorted(
    glob.glob(os.path.join(root_path, "*", "*", "HHZ")))
for station_path in station_dirs:
    process_station(station_path)
print("\nDONE: mean-removed daily ACF saved for all stations.")
