import os
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ruamel.yaml import YAML
import librosa
import numpy as np
import tempfile
import wave
import matplotlib.pyplot as plt
import librosa.display
import pyworld as world
import soundfile as sf
import struct
import customtkinter as ctk
import platform

base_path = os.path.dirname(os.path.abspath(__file__))
ffmpeg_folder = os.path.join(base_path, "ffmpeg")

os.environ["PATH"] = ffmpeg_folder + os.pathsep + os.environ["PATH"]
from pydub import AudioSegment

print("FFMPEG folder injected:", ffmpeg_folder)

yaml = YAML()
yaml.preserve_quotes = True

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class UTAUResamplerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Batch Resampler GUI")
        
        self.config_path = "config.yaml"
        self.config = self.load_config()
        self.audio_files = []
        self.batch_checkboxes = {}
        self.geometry("600x600")

        # Variables 
        self.resampler_dir_var = ctk.StringVar(value=self.config.get("resampler_directory", ""))
        self.output_dir_var = ctk.StringVar(value=self.config.get("output_directory", ""))
        self.resampler_var = ctk.StringVar(value=self.config.get("default_resampler", ""))
        
        self.flags_var = ctk.StringVar(value="")
        self.pitch_note_var = ctk.StringVar(value="C4")
        self.volume_var = ctk.StringVar(value="100")
        self.modulation_var = ctk.StringVar(value="0")
        self.threads_var = ctk.StringVar(value="4")

        self.setup_ui()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.load(f) or {}
            except: pass
        return {"resampler_directory": "", "default_resampler": "", "output_directory": ""}

    def update_config_file(self, key, value):
        self.config[key] = value
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def setup_ui(self):
        self.main_container = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # File Selection Section
        self.top_frame = ctk.CTkFrame(self.main_container)
        self.top_frame.pack(fill="x", padx=10, pady=5)
        self.top_frame.grid_columnconfigure(1, weight=1) 

        ctk.CTkLabel(self.top_frame, text="Audio Files:").grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.audio_textbox = ctk.CTkTextbox(self.top_frame, height=100)
        self.audio_textbox.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.audio_textbox.configure(state="disabled")

        btn_frame = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=10, sticky="n") # Sticky "n" keeps it at the top
        ctk.CTkButton(btn_frame, text="Add Files", width=120, command=self.add_audio_files).pack(pady=5)
        ctk.CTkButton(btn_frame, text="Clear", width=120, fg_color="transparent", border_width=1, command=self.clear_audio_files).pack(pady=5)
        ctk.CTkLabel(self.top_frame, text="Resampler Dir:").grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.resampler_entry = ctk.CTkEntry(self.top_frame, textvariable=self.resampler_dir_var)
        self.resampler_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(self.top_frame, text="Browse", width=120, command=self.select_resampler_dir).grid(row=1, column=2, padx=10, pady=5)
        
        # Tabs Section
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="x", padx=10, pady=5)
        self.single_tab = self.tabview.add("Single Resample")
        self.batch_tab = self.tabview.add("Batch Resamplers")

        # Single UI
        ctk.CTkLabel(self.single_tab, text="Select Resampler:").pack(pady=5)
        self.resampler_combo = ctk.CTkComboBox(self.single_tab, variable=self.resampler_var, values=self.get_resamplers(), width=400, command=self.save_resampler_config)
        self.resampler_combo.pack(pady=5)

        # Batch UI
        ctk.CTkLabel(self.batch_tab, text="Select Multiple Resamplers:").pack(pady=5)
        self.batch_scroll = ctk.CTkScrollableFrame(self.batch_tab, height=100)
        self.batch_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.refresh_batch_resamplers()

        # Parameters Section
        self.param_frame = ctk.CTkFrame(self.main_container)
        self.param_frame.pack(fill="x", padx=10, pady=5)
        self.param_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Title: Center aligned across all columns
        ctk.CTkLabel(self.param_frame, text="Processing Parameters", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=4, pady=10, sticky="ew"
        )

        # Row 1: Flags and Threads
        ctk.CTkLabel(self.param_frame, text="Flags:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.flags_var, width=140).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Threads:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.threads_var, width=60).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Row 2: Pitch and Follow Pitch
        ctk.CTkLabel(self.param_frame, text="Pitch:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.pitch_entry = ctk.CTkEntry(self.param_frame, textvariable=self.pitch_note_var, width=140)
        self.pitch_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.follow_pitch_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.param_frame, text="Follow Input Pitch", variable=self.follow_pitch_var, command=self.toggle_pitch_entry).grid(
            row=2, column=2, columnspan=2, padx=5, pady=5, sticky="w"
        )

        # Row 3: Volume and Modulation
        ctk.CTkLabel(self.param_frame, text="Volume:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.volume_var, width=140).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Modulation:").grid(row=3, column=2, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.modulation_var, width=140).grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # Row 4: Extra Options (Checkboxes)
        self.gen_spec_var = ctk.BooleanVar(value=False)
        self.generate_frq_var = ctk.BooleanVar(value=False)
        self.only_frq_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(self.param_frame, text="Spectrograms", variable=self.gen_spec_var).grid(row=4, column=1, pady=10, sticky="w")
        ctk.CTkCheckBox(self.param_frame, text="Harvest .frq", variable=self.generate_frq_var).grid(row=4, column=2, pady=10, sticky="w")
        ctk.CTkCheckBox(self.param_frame, text="Only .frq", variable=self.only_frq_var).grid(row=4, column=3, pady=10, sticky="w")

        # Output Section
        self.out_f = ctk.CTkFrame(self.main_container)
        self.out_f.pack(fill="x", padx=10, pady=5)
        self.out_f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.out_f, text="Output Dir:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.output_entry = ctk.CTkEntry(self.out_f, textvariable=self.output_dir_var)
        self.output_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(self.out_f, text="Browse", width=120, command=self.select_output_dir).grid(row=0, column=2, padx=10, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.main_container)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.set(0)

        self.start_btn = ctk.CTkButton(self.main_container, text="START RESAMPLING", height=50, fg_color="#28a745", hover_color="#218838", font=("Arial", 16, "bold"), command=self.run_process_thread)
        self.start_btn.pack(pady=20, padx=20, fill="x")
        
    # Logic
    def save_resampler_config(self, choice):
        self.update_config_file("default_resampler", choice)
              
    def select_resampler_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.resampler_dir_var.set(d)
            self.update_config_file("resampler_directory", d)
            self.resampler_combo.configure(values=self.get_resamplers())
            self.refresh_batch_resamplers()

    def select_output_dir(self):
        d = filedialog.askdirectory()
        if d: 
            self.output_dir_var.set(d)
            self.update_config_file("output_directory", d)

    def toggle_pitch_entry(self):
        state = "disabled" if self.follow_pitch_var.get() else "normal"
        self.pitch_entry.configure(state=state)

    def get_resamplers(self):
        base_dir = self.resampler_dir_var.get()
        resampler_list = []
        
        if os.path.isdir(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.lower().endswith((".exe", ".bat")):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, base_dir)
                        resampler_list.append(relative_path)
        
        return sorted(resampler_list)

    def refresh_batch_resamplers(self):
        for widget in self.batch_scroll.winfo_children():
            widget.destroy()
        self.batch_checkboxes = {}
        for r in self.get_resamplers():
            var = ctk.BooleanVar()
            cb = ctk.CTkCheckBox(self.batch_scroll, text=r, variable=var)
            cb.pack(anchor="w", pady=2)
            self.batch_checkboxes[r] = var

    def add_audio_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")])
        if files:
            for f in files:
                if f not in self.audio_files:
                    self.audio_files.append(f)
            self.update_audio_display()

    def clear_audio_files(self):
        self.audio_files = []
        self.update_audio_display()

    def update_audio_display(self):
        self.audio_textbox.configure(state="normal")
        self.audio_textbox.delete("1.0", "end")
        for f in self.audio_files:
            self.audio_textbox.insert("end", os.path.basename(f) + "\n")
        self.audio_textbox.configure(state="disabled")

    # Core Logic 
    def note_to_hz(self, note_name):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_name.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#').replace('Ab', 'G#').replace('Bb', 'A#')
        try:
            name = note_name[:-1]
            octave = int(note_name[-1])
            n = notes.index(name)
            return 440.0 * (2.0 ** ((n + (octave - 4) * 12 - 9) / 12.0))
        except: return 440.0

    def detect_pitch(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) == 0: return "C4"
            f0 = librosa.yin(y_trimmed[:int(sr*1.0)], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                avg_f0 = np.nanmedian(f0)
                note = librosa.midi_to_note(int(round(librosa.hz_to_midi(avg_f0))))
                return note.replace('♯', '#').replace('♭', 'b')
        except Exception as e:
            print(f"Pitch detection failed: {e}")
        return "C4"

    def save_spectrogram(self, audio_path):
        try:
            plt.figure(figsize=(10, 4))
            y, sr = librosa.load(audio_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
            plt.title(f"Spectrogram: {os.path.basename(audio_path)}")
            plt.tight_layout()
            plt.savefig(os.path.splitext(audio_path)[0] + "_spec.png")
            plt.close()
        except: pass

    def generate_harvest_frq(self, input_wav, target_out=None):
        try:
            x, fs = sf.read(input_wav)
            if x.ndim > 1: x = x[:, 0]
            hop = 256
            f0, t = world.harvest(x, fs, f0_ceil=880, frame_period=1000*hop/fs)
            sp = world.cheaptrick(x, f0, t, fs)
            amp = np.mean(np.sqrt(sp), axis=1)
            base_f0 = np.median(f0[f0 > 0]) if any(f0 > 0) else 440.0
            
            frq_path = os.path.splitext(target_out if target_out else input_wav)[0] + "_wav.frq"
            with open(frq_path, 'wb') as f:
                f.write(b'FREQ0003')
                f.write(struct.pack('i', hop))
                f.write(struct.pack('d', base_f0))
                f.write(bytes(16))
                f.write(struct.pack('i', f0.shape[0]))
                for i in range(f0.shape[0]):
                    f.write(struct.pack('2d', f0[i], amp[i]))
            return frq_path
        except: return None

    # Processing Thread 
    def run_process_thread(self):
        threading.Thread(target=self.execute_resampling, daemon=True).start()

    def execute_resampling(self):
        if not self.audio_files or not self.output_dir_var.get():
            messagebox.showerror("Error", "Missing files or output directory")
            return
        
        self.start_btn.configure(state="disabled", text="PROCESSING...")
        errors = []
        
        # Color codes
        BLUE, YELLOW, GREEN, RED, CYAN, RESET, BOLD = "\033[94m", "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[0m", "\033[1m"

        mode = self.tabview.get()
        res_dir = self.resampler_dir_var.get()
        resamplers = []
        if mode == "Single Resample":
            if self.resampler_var.get(): resamplers.append(os.path.join(res_dir, self.resampler_var.get()))
        else:
            for name, var in self.batch_checkboxes.items():
                if var.get(): resamplers.append(os.path.join(res_dir, name))

        if not resamplers:
            self.after(0, lambda: self.start_btn.configure(state="normal", text="START RESAMPLING"))
            return

        temp_map = {} # Original Path -> Normalized Temp Path
        pitch_map = {}
        specific_temp_dir = os.path.join(base_path, "cache_temp")
        os.makedirs(specific_temp_dir, exist_ok=True)

        # STAGE 1: Normalization & Global FRQ (Run once per file)
        print(f"\n{BLUE}{BOLD}=== STAGE 1: PRE-PROCESSING AUDIO ==={RESET}")
        for audio in self.audio_files:
            file_name = os.path.basename(audio)
            print(f"{CYAN}{BOLD}[ANALYZING]{RESET} {file_name}")
            
            try:
                seg = AudioSegment.from_file(audio)
                
                # Check UTAU Compliance
                is_wav = audio.lower().endswith(".wav")
                is_mono = seg.channels == 1
                is_16bit = seg.sample_width == 2
                is_44k = seg.frame_rate == 44100

                if not (is_wav and is_mono and is_16bit and is_44k):
                    reasons = []
                    if not is_wav: reasons.append("Format not WAV")
                    if not is_mono: reasons.append(f"Stereo ({seg.channels} channels)")
                    if not is_16bit: reasons.append(f"Bit-depth ({seg.sample_width * 8}-bit)")
                    if not is_44k: reasons.append(f"Sample rate ({seg.frame_rate}Hz)")
                    
                    print(f"  {YELLOW}>> Normalizing: {', '.join(reasons)}...{RESET}")
                    
                    t_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=specific_temp_dir, delete=False)
                    normalized_path = t_file.name
                    t_file.close()
                    
                    # Apply UTAU Standards
                    seg.set_channels(1).set_sample_width(2).set_frame_rate(44100).export(normalized_path, format="wav")
                    temp_map[audio] = normalized_path
                    print(f"  {GREEN}>> Temporary UTAU-compliant WAV created.{RESET}")
                else:
                    print(f"  {GREEN}>> Already UTAU-compliant. Using original file.{RESET}")
                    temp_map[audio] = audio

                # Pitch Detection (once per file)
                detected_pitch = self.detect_pitch(temp_map[audio]) if self.follow_pitch_var.get() else self.pitch_note_var.get()
                pitch_map[audio] = detected_pitch
                print(f"  {BOLD}>> Pitch:{RESET} {detected_pitch}")

                # Harvest FRQ (once per file)
                if self.generate_frq_var.get():
                    print(f"  {CYAN}>> Generating Harvest FRQ...{RESET}")
                    self.generate_harvest_frq(temp_map[audio])

            except Exception as e:
                print(f"{RED}[PRE-PROCESS FAILED] {file_name}: {e}{RESET}")

        # STAGE 2: Multi-Resampler Rendering
        print(f"\n{BLUE}{BOLD}=== STAGE 2: RENDERING WITH RESAMPLERS ==={RESET}")
        tasks = []
        for audio in self.audio_files:
            if audio not in temp_map: continue
            for res_path in resamplers:
                res_clean = os.path.splitext(os.path.basename(res_path))[0]
                out_folder = self.output_dir_var.get() if len(resamplers) == 1 else os.path.join(self.output_dir_var.get(), res_clean)
                os.makedirs(out_folder, exist_ok=True)
                out_file = os.path.join(out_folder, f"{os.path.splitext(os.path.basename(audio))[0]}.wav")
                
                # Check for Only FRQ mode
                if self.only_frq_var.get():
                    self.generate_harvest_frq(temp_map[audio], target_out=out_file)
                    continue

                tasks.append({
                    "res_exe": res_path,
                    "audio_in": temp_map[audio],
                    "audio_out": out_file,
                    "pitch": pitch_map[audio]
                })

        if tasks:
            total = len(tasks)
            try: num_threads = int(self.threads_var.get() or 1)
            except: num_threads = 1
            
            with ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
                futures = {executor.submit(self.run_resampler, t): t for t in tasks}
                
                for i, future in enumerate(futures):
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"File {os.path.basename(futures[future]['audio_out'])}: {str(e)}")
                    self.after(0, lambda v=(i+1)/total: self.progress_bar.set(v))

        # STAGE 3: Final Cleanup
        print(f"\n{BLUE}{BOLD}=== STAGE 3: CLEANUP ==={RESET}")
        for orig, temp in temp_map.items():
            if orig != temp and os.path.exists(temp):
                try: 
                    os.remove(temp)
                    print(f"Removed temp: {os.path.basename(temp)}")
                except: pass
        
        self.after(0, lambda: self.finish_job(errors))

    def run_resampler(self, t):
        # ANSI Color Codes
        YELLOW, GREEN, RED, CYAN, RESET, BOLD = "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[0m", "\033[1m"
        
        try:
            with wave.open(t["audio_in"], 'r') as f:
                raw_len = int((f.getnframes() / float(f.getframerate())) * 1000)
                length, cons = str(max(0, raw_len)), str(max(0, raw_len - 100))

            # arguements
            if "fader2" in os.path.basename(t["res_exe"]).lower():
                # fader2: <in> <in2> <out> <pitch_hz> <length_ms> <ratio>
                cmd = [
                    t["res_exe"], 
                    t["audio_in"], 
                    "null", 
                    t["audio_out"], 
                    str(float(self.note_to_hz(t["pitch"]))), 
                    length, 
                    "0.0"
                ]
            else:
                # Standard UTAU: <in> <out> <pitch> <vel> <flags> <offset> <length> <consonant> <cutoff> <vol> <mod> <tempo> <bend>
                cmd = [
                    t["res_exe"], 
                    t["audio_in"], 
                    t["audio_out"], 
                    t["pitch"], 
                    "100", 
                    self.flags_var.get() or "", 
                    "0", 
                    length, 
                    cons, 
                    "0", 
                    self.volume_var.get(), 
                    self.modulation_var.get(), 
                    "!120", 
                    "AA#1000#"
                ]

            print(f"{YELLOW}{BOLD}[EXEC] {os.path.basename(t['res_exe'])} -> {os.path.basename(t['audio_out'])}{RESET}")
            print(f"{BOLD}Command Arguments:{RESET}")
            for i, arg in enumerate(cmd):
                print(f"  {CYAN}[{i}]{RESET} {arg}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                print(f"{BOLD}Output:{RESET} {result.stdout.strip()}")
            
            print(f"{GREEN}[SUCCESS] Rendered: {os.path.basename(t['audio_out'])}{RESET}")
            
            if self.gen_spec_var.get(): 
                self.save_spectrogram(t["audio_out"])
                
        except subprocess.CalledProcessError as e:
            print(f"{RED}[FAILED] Resampler returned error code {e.returncode}{RESET}")
            if e.stderr:
                print(f"{RED}Error Details: {e.stderr.strip()}{RESET}")
        except Exception as e:
            print(f"{RED}[ERROR] {os.path.basename(t['audio_out'])}: {e}{RESET}")

    def finish_job(self, errors):
        self.start_btn.configure(state="normal", text="START RESAMPLING")
        self.progress_bar.set(0) # Reset progress bar

        if not errors:
            messagebox.showinfo("Success", "All files processed successfully!")
        else:
            error_msg = "\n".join(errors[:5]) # Show first 5 errors to avoid a massive popup
            if len(errors) > 5:
                error_msg += f"\n...and {len(errors) - 5} more."
            
            messagebox.showerror("Processing Errors", f"Completed with errors:\n\n{error_msg}")

if __name__ == "__main__":
    app = UTAUResamplerGUI()
    app.mainloop()