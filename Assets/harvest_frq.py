import numpy as np
import soundfile as sf
import pyworld as world
import os
import struct

def calculate_base_frq(f0, f0_ceil):
    n = len(f0)
    if n < 2:
        return np.median(f0) if n > 0 else 440.0
    
    avg_frq, tally = 0, 0
    f0_floor = 40.0 

    for i in range(n):
        if f0_floor <= f0[i] <= f0_ceil:
            if i < 1:
                q = f0[i+1] - f0[i]
            elif i == n - 1:
                q = f0[i] - f0[i-1]
            else:
                q = (f0[i+1] - f0[i-1]) / 2
            
            weight = 2 ** (-q * q)
            avg_frq += f0[i] * weight
            tally += weight

    return (avg_frq / tally) if tally > 0 else 440.0

def generate_harvest_frq(input_wav, target_out=None):
    try:
        x, fs = sf.read(input_wav)
        if x.ndim > 1: x = np.mean(x, axis=1)
        
        hop = 256
        f0_ceil = 1100
        
        # Heavy WORLD math (Harvest)
        f0, t = world.harvest(x, fs, f0_ceil=f0_ceil, frame_period=1000 * hop / fs)
        
        # Amplitude calculation
        window_size = hop * 2
        energy = np.array([
            np.sqrt(np.mean(x[int(ti*fs) : int(ti*fs) + window_size]**2)) 
            if int(ti*fs) < len(x) else 0 
            for ti in t
        ])
        
        base_f0 = calculate_base_frq(f0, f0_ceil)
        frq_path = os.path.splitext(target_out if target_out else input_wav)[0] + "_wav.frq"
        
        frq_data = np.stack((f0, energy), axis=-1).astype(np.float64)

        with open(frq_path, 'wb') as f:
            f.write(b'FREQ0003')
            f.write(struct.pack('i', hop))
            f.write(struct.pack('d', base_f0))
            f.write(bytes(16))
            f.write(struct.pack('i', len(f0)))
            f.write(frq_data.tobytes())
            
        return frq_path
    except Exception as e:
        print(f"Error in worker: {e}")
        return None