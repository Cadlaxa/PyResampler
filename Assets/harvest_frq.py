import numpy as np
import soundfile as sf
import pyworld as world
import os
import struct
import concurrent.futures

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

# Helper function for parallel processing
def _harvest_worker(x_chunk, fs, f0_ceil, frame_period, pad_frames, target_f0_len):
    f0_chunk, _ = world.harvest(x_chunk, fs, f0_ceil=f0_ceil, frame_period=frame_period)
    return f0_chunk[pad_frames : pad_frames + target_f0_len]

def generate_harvest_frq(input_wav, target_out=None, user_threads=1):
    try:
        x, fs = sf.read(input_wav)
        if x.ndim > 1: x = np.mean(x, axis=1)
        x = x.astype(np.float64) 
        
        hop = 256
        f0_ceil = 1100.0
        frame_period = 1000.0 * hop / fs
        
        frames_per_chunk = 2000  
        pad_frames = 100         
        
        chunk_samples = frames_per_chunk * hop
        pad_samples = pad_frames * hop
        len_x = len(x)
        
        futures = []
        
        # Safely parse user threads, fallback to 1 if it fails
        try:
            max_threads = max(1, int(user_threads))
        except (ValueError, TypeError):
            max_threads = 1
            
        # --- THE FIX: MULTI-THREADING ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            for i in range(0, len_x, chunk_samples):
                start_idx = i - pad_samples
                end_idx = i + chunk_samples + pad_samples
                
                pad_left = max(0, -start_idx)
                pad_right = max(0, end_idx - len_x)
                
                safe_start = max(0, start_idx)
                safe_end = min(len_x, end_idx)
                x_chunk = x[safe_start:safe_end]
                
                if pad_left > 0 or pad_right > 0:
                    x_chunk = np.pad(x_chunk, (pad_left, pad_right), mode='constant')
                
                target_f0_len = min(frames_per_chunk, (len_x - i) // hop + 1)
                
                # Send the chunk to a background thread
                futures.append(executor.submit(
                    _harvest_worker, x_chunk, fs, f0_ceil, frame_period, pad_frames, target_f0_len
                ))

        # Retrieve results in order and stitch
        f0_segments = [f.result() for f in futures]
        f0 = np.concatenate(f0_segments)
        
        # Optimized Energy Math (Pre-allocated array prevents memory spikes)
        t = np.arange(len(f0)) * (hop / fs)
        energy = np.zeros(len(t), dtype=np.float64)
        window_size = hop * 2
        
        for i_t, ti in enumerate(t):
            s_idx = int(ti * fs)
            e_idx = min(s_idx + window_size, len_x)
            if s_idx < len_x:
                energy[i_t] = np.sqrt(np.mean(x[s_idx:e_idx]**2))

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
        print(f"\033[91m[ERROR] Harvest Worker Crash: {e}\033[0m")
        return None

def slice_master_frq(master_frq_path, chunk_wav_path, chunk_start_ms):
    try:
        x, fs = sf.read(chunk_wav_path)
        if x.ndim > 1: x = np.mean(x, axis=1)

        with open(master_frq_path, 'rb') as f:
            header = f.read(40)
            global_hop = struct.unpack('i', header[8:12])[0]
            global_base_f0 = struct.unpack('d', header[12:20])[0]
            
            data = f.read()
            n_frq_frames = len(data) // 16
            frames = struct.unpack('<' + 'd' * (n_frq_frames * 2), data)
            f0_raw = np.array(frames[0::2], dtype=np.float64)
            energy_raw = np.array(frames[1::2], dtype=np.float64)

        # --- THE FIX: DRIFT-PROOF ALIGNMENT ---
        start_sample = int(round((chunk_start_ms / 1000.0) * fs))
        start_frame = int(round(start_sample / global_hop))
        
        target_frames = len(x) // global_hop + 1 
        end_frame = start_frame + target_frames

        f0_slice = f0_raw[start_frame:end_frame]
        energy_slice = energy_raw[start_frame:end_frame]

        if len(f0_slice) < target_frames:
            pad_len = target_frames - len(f0_slice)
            f0_slice = np.pad(f0_slice, (0, pad_len), mode='edge')
            energy_slice = np.pad(energy_slice, (0, pad_len), mode='constant')

        frq_path = os.path.splitext(chunk_wav_path)[0] + "_wav.frq"
        frq_data = np.stack((f0_slice, energy_slice), axis=-1).astype(np.float64)

        with open(frq_path, 'wb') as f:
            f.write(b'FREQ0003')
            f.write(struct.pack('i', global_hop))
            f.write(struct.pack('d', global_base_f0)) 
            f.write(bytes(16))
            f.write(struct.pack('i', len(f0_slice)))
            f.write(frq_data.tobytes())
            
        return frq_path
    except Exception as e:
        print(f"\033[91m[ERROR] Slicing FRQ: {e}\033[0m")
        return None