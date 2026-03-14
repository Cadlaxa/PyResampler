import os
import struct
import numpy as np
import soundfile as sf
import librosa
import pyworld as pw
import threading
from ruamel.yaml import YAML

_SESSION_CACHE = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock() 

def get_ort_session(model_path):
    import onnxruntime as ort
    with _CACHE_LOCK:
        if model_path not in _SESSION_CACHE:
            print(f"  [ONNX] Loading model into memory: {os.path.basename(model_path)}...")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            try:
                print(f"  [ONNX] Warming up model memory...")
                inp_names = [i.name for i in session.get_inputs()]
                dummy_inputs = {}
                for name in inp_names:
                    if 'mel' in name.lower() or 'spec' in name.lower():
                        dummy_inputs[name] = np.zeros((1, 10, 128), dtype=np.float32)
                    elif 'f0' in name.lower() or 'pitch' in name.lower():
                        dummy_inputs[name] = np.full((1, 10), 440.0, dtype=np.float32)
                session.run(None, dummy_inputs)
            except Exception as e:
                pass
                
            _SESSION_CACHE[model_path] = session
            
        return _SESSION_CACHE[model_path]

def note_to_hz(note_name):
    try:
        return float(note_name)
    except ValueError:
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_name.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#').replace('Ab', 'G#').replace('Bb', 'A#')
        try:
            name = note_name[:-1]
            octave = int(note_name[-1])
            n = notes.index(name)
            return 440.0 * (2.0 ** ((n + (octave - 4) * 12 - 9) / 12.0))
        except: 
            return 440.0

def run_onnx_inference(model_path, input_path, output_path, pitch, length_ms, volume=100.0, modulation=100.0, original_path=None, chunk_start_ms=0.0, speed_multiplier=1.0):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    yaml = YAML(typ='safe')

    model_dir = os.path.dirname(model_path)
    yaml_path = os.path.join(model_dir, "vocoder.yaml")
    config = {}
    
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f) or {}
        except Exception as e:
            print(f"    \033[93m[WARNING]\033[0m Failed to parse {os.path.basename(yaml_path)}. Falling back to defaults: {e}")
    else:
        print(f"    \033[93m[WARNING]\033[0m No YAML found for {os.path.basename(model_path)}. Falling back to defaults.")

    TARGET_SR = config.get("sample_rate", 44100)
    HOP_LENGTH = config.get("hop_size", 512)
    WIN_LENGTH = config.get("win_size", 2048)
    N_FFT = config.get("fft_size", 2048)
    N_MELS = config.get("num_mel_bins", 128)
    FMIN = config.get("mel_fmin", 40)
    
    fmax_raw = config.get("mel_fmax", 16000)
    FMAX = 16000 if fmax_raw is None else fmax_raw

    MEL_BASE = config.get("mel_base", "10") 
    MEL_SCALE = config.get("mel_scale", "slaney")
    PITCH_CONTROLLABLE = config.get("pitch_controllable", True)

    if sample_rate != TARGET_SR:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    # 1. Amplitude Scaler 
    wave_max = np.max(np.abs(audio_data))
    if wave_max >= 0.5:
        audio_scale = 0.5 / wave_max
        audio_data = audio_data * audio_scale
    else:
        audio_scale = 1.0

    # 2. Mel Spectrogram 
    pad_size = int((N_FFT - HOP_LENGTH) / 2)
    audio_padded = np.pad(audio_data, (pad_size, pad_size), mode='reflect')
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_padded, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, 
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=False,
        power=1.0 
    )
    mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
    
    mel_tensor = mel_spec.T
    mel_tensor = np.expand_dims(mel_tensor, axis=0).astype(np.float32)
    num_frames = mel_tensor.shape[1]

    target_hz = note_to_hz(pitch)
    
    # 3. Pitch Generation 
    if modulation == 0.0:
        f0_tensor_1d = np.full(num_frames, target_hz, dtype=np.float32)
    else:
        original_f0 = None
        found_frq_path = None
        is_original_frq = False
        
        input_dir = os.path.dirname(input_path)
        input_base = os.path.splitext(os.path.basename(input_path))[0]
        
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.startswith(input_base) and f.lower().endswith('.frq'):
                    found_frq_path = os.path.join(input_dir, f)
                    is_original_frq = False
                    break
        
        if not found_frq_path and original_path:
            orig_dir = os.path.dirname(original_path)
            orig_base = os.path.splitext(os.path.basename(original_path))[0]
            if os.path.exists(orig_dir):
                for f in os.listdir(orig_dir):
                    if f.startswith(orig_base) and f.lower().endswith('.frq'):
                        found_frq_path = os.path.join(orig_dir, f)
                        is_original_frq = True
                        break

        if found_frq_path:
            try:
                with open(found_frq_path, 'rb') as f:
                    header = f.read(40)
                    if header.startswith(b'FREQ0003'):
                        data = f.read()
                        n_frq_frames = len(data) // 16
                        if n_frq_frames > 0:
                            global_base_f0 = struct.unpack('d', header[12:20])[0]
                            
                            frames = struct.unpack('<' + 'd' * (n_frq_frames * 2), data)
                            f0_raw = np.array(frames[0::2], dtype=np.float64)
                            
                            if is_original_frq:
                                t_frq = np.arange(len(f0_raw)) * (256.0 / 44100.0)
                                t_chunk = np.arange(num_frames) * (HOP_LENGTH / TARGET_SR)
                                t_chunk_in_orig = ((chunk_start_ms / 1000.0) + t_chunk) * speed_multiplier
                                original_f0 = np.interp(t_chunk_in_orig, t_frq, f0_raw)
                            else:
                                original_f0 = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(f0_raw)), f0_raw)
            except Exception as e:
                pass

        if original_f0 is None or len(original_f0) == 0:
            _audio_for_pw = audio_data / audio_scale 
            _f0, _t = pw.dio(_audio_for_pw.astype(np.float64), TARGET_SR, frame_period=(HOP_LENGTH/TARGET_SR)*1000)
            original_f0 = pw.stonemask(_audio_for_pw.astype(np.float64), _f0, _t, TARGET_SR)
            if len(original_f0) != num_frames:
                original_f0 = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(original_f0)), original_f0)
            
        voiced_indices = original_f0 > 0
        f0_tensor_1d = np.zeros_like(original_f0, dtype=np.float32)
        
        if np.any(voiced_indices):
            f0_midi = np.zeros_like(original_f0)
            f0_midi[voiced_indices] = 12 * np.log2(original_f0[voiced_indices] / 440.0) + 69
            target_midi = 12 * np.log2(target_hz / 440.0) + 69
            
            if 'global_base_f0' in locals() and global_base_f0 > 0:
                median_midi = 12 * np.log2(global_base_f0 / 440.0) + 69
            else:
                median_midi = np.median(f0_midi[voiced_indices])
            
            mod_factor = modulation / 100.0
            f0_midi[voiced_indices] = target_midi + (f0_midi[voiced_indices] - median_midi) * mod_factor
            f0_tensor_1d[voiced_indices] = 440.0 * (2.0 ** ((f0_midi[voiced_indices] - 69) / 12.0))
        
        f0_tensor_1d[~voiced_indices] = target_hz

    f0_tensor = np.expand_dims(f0_tensor_1d, axis=0).astype(np.float32)

    # 4. Inference (USING THE INFERENCE LOCK) 
    session = get_ort_session(model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    
    inputs = {}
    for name in input_names:
        if 'mel' in name.lower() or 'spec' in name.lower():
            inputs[name] = mel_tensor
        elif 'f0' in name.lower() or 'pitch' in name.lower():
            inputs[name] = f0_tensor
            
    with _INFERENCE_LOCK:
        output = session.run(None, inputs)
        
    output_audio = output[0].squeeze()

    # 5. Final Volume & Limiter 
    output_audio = output_audio / audio_scale 
    base_boost = 0.8
    output_audio = output_audio * base_boost
    volume_factor = volume / 100.0
    output_audio = output_audio * volume_factor
    output_audio = np.clip(output_audio, -0.99, 0.99)

    target_samples = int((length_ms / 1000.0) * TARGET_SR)
    if len(output_audio) > target_samples:
        output_audio = output_audio[:target_samples]
    elif len(output_audio) < target_samples:
        pad_amount = target_samples - len(output_audio)
        output_audio = np.pad(output_audio, (0, pad_amount), mode='constant')

    sf.write(output_path, output_audio, TARGET_SR, subtype='PCM_16')