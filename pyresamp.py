import os, sys, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ruamel.yaml import YAML, version
import librosa, tempfile, wave, matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import pyworld as world
import soundfile as sf
import customtkinter as ctk
import struct, platform, webbrowser, requests, shutil, re, pyglet, ctypes
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path as P
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

base_path = os.path.dirname(os.path.abspath(__file__))
yaml = YAML()
yaml.preserve_quotes = True

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("Assets/Theme/orange.json")

ASSETS = P('./Assets')
FONTS = P(ASSETS, 'Fonts')

def setup_ffmpeg():
    current_os = platform.system()
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    os_map = {
        "Windows": "Assets/Ffmpeg/win",
        "Darwin": "Assets/Ffmpeg/mac",
        "Linux": "Assets/Ffmpeg/linux"
    }
    
    folder_name = os_map.get(current_os)
    if not folder_name:
        print(f"[{current_os}] OS not recognized for local bundling.")
        return

    ffmpeg_folder = os.path.join(base_path, folder_name)
    if os.path.exists(ffmpeg_folder):
        os.environ["PATH"] = ffmpeg_folder + os.pathsep + os.environ["PATH"]
        
        if current_os in ["Darwin", "Linux"]:
            for bin_file in ["ffmpeg", "ffprobe"]:
                bin_path = os.path.join(ffmpeg_folder, bin_file)
                if os.path.exists(bin_path):
                    # Standard chmod +x equivalent
                    st = os.stat(bin_path)
                    os.chmod(bin_path, st.st_mode | os.stat.S_IEXEC)
        
        print(f"[{current_os}] FFmpeg loaded from: {ffmpeg_folder}")
    else:
        print(f"[{current_os}] Folder {folder_name} missing. Using system default.\n")
setup_ffmpeg()
from pydub import AudioSegment

class UTAUResamplerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        icon_path = P(__file__).parent / "Assets" / "icon.ico"
        self.iconbitmap(str(icon_path))

        self.title("PyResampler - Batch Resampling GUI")
        self.version = "0.0.6"
        self.config_path = "config.yaml"
        self.yaml = YAML()
        self.audio_files = []
        self.batch_checkboxes = {}
        self.geometry("700x600")
        self.specific_temp_dir = os.path.join(base_path, "Cache_temp")
        print(f"PyResampler Verion: {self.version}")
        self.trim_data = {}

        # Variables 
        self.flags_var = ctk.StringVar(value="")
        self.pitch_note_var = ctk.StringVar(value="C4")
        self.volume_var = ctk.StringVar(value="100")
        self.modulation_var = ctk.StringVar(value="100")
        self.threads_var = ctk.StringVar(value="4")
        self.gen_spec_var = ctk.BooleanVar(value=False)
        self.generate_frq_var = ctk.BooleanVar(value=False)
        self.only_frq_var = ctk.BooleanVar(value=False)
        self.follow_pitch_var = ctk.BooleanVar(value=False)
        self.speed_var = ctk.DoubleVar(value=0.0)
        self.load_config()

        self.setup_ui()
        self.toggle_pitch_entry()
        self.check_for_updates()
    
    def clear_cache(self):
        temp_dir = self.specific_temp_dir
        
        if not os.path.exists(temp_dir):
            messagebox.showinfo("Info", "No temp folder found.")
            return
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete ALL files in the temp folder?")
        
        if confirm:
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
                messagebox.showinfo("Success", "Temp folder cleared successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not clear temp folder: {e}")
    
    def check_for_updates(self):
        url = "https://raw.githubusercontent.com/Cadlaxa/PyResampler/main/version.txt"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                raw_text = response.text.encode('utf-8').decode('utf-8-sig').strip()
                latest_version = raw_text.replace("version:", "").strip()
                
                current_version = str(self.version).strip()
                if latest_version > current_version:
                    user_choice = messagebox.askyesno(
                        "Update Available", 
                        f"A new version ({latest_version}) is available!\n\n"
                        f"Current version: {current_version}\n"
                        "Would you like to go to the GitHub page to download the update?"
                    )
                    if user_choice:
                        webbrowser.open("https://github.com/Cadlaxa/PyResampler")
                else:
                    print("Script is up to date.")
                
        except Exception as e:
            print(f"Update check failed: {e}")

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    # Using 'or {}' ensures it's never None
                    data = self.yaml.load(f)
                    self.config = data if data is not None else {}
            except Exception as e:
                print(f"Error: {e}")
                self.config = {}
        else:
            self.config = {}

        self.follow_pitch_var.set(self.config.get("follow_pitch", False))
        self.gen_spec_var.set(self.config.get("generate_spectrogram", False))
        self.generate_frq_var.set(self.config.get("generate_frq", False))
        self.only_frq_var.set(self.config.get("only_frq", False))
        
        self.volume_var.set(str(self.config.get("volume", "100")))
        self.modulation_var.set(str(self.config.get("modulation", "100")))
        self.flags_var.set(str(self.config.get("flags", "")))
        self.threads_var.set(str(self.config.get("threads", "4")))
        self.pitch_note_var.set(str(self.config.get("pitch_note", "C4")))
        
        self.speed_var.set(float(self.config.get("speed", 0.0)))
        self.resampler_dir_var = ctk.StringVar(value=self.config.get("resampler_directory", ""))
        self.output_dir_var = ctk.StringVar(value=self.config.get("output_directory", ""))
        self.resampler_var = ctk.StringVar(value=self.config.get("default_resampler", ""))
        return self.config

    def save_config(self, key=None, value=None):
        if self.config is None:
            self.config = {}

        if key is not None:
            self.config[key] = value

        # Sync all GUI variables into the config dictionary
        gui_settings = {
            "follow_pitch": self.follow_pitch_var.get(),
            "volume": self.volume_var.get(),
            "modulation": self.modulation_var.get(),
            "generate_spectrogram": self.gen_spec_var.get(),
            "generate_frq": self.generate_frq_var.get(),
            "only_frq": self.only_frq_var.get(),
            "flags": self.flags_var.get(),
            "threads": self.threads_var.get(),
            "pitch_note": self.pitch_note_var.get(),
            "speed": self.speed_var.get(),
            "resampler_directory": self.resampler_dir_var.get(),
            "output_directory": self.output_dir_var.get()
        }
        self.config.update(gui_settings)

        try:
            with open(self.config_path, 'w') as f:
                self.yaml.dump(self.config, f)
        except PermissionError:
            print("\033[91m[ERROR] Config.yaml is locked/open in another app!\033[0m")
        except Exception as e:
            print(f"\033[91m[ERROR] Save failed: {e}\033[0m")
    
    def update_speed_label(self, value):
        rounded_val = round(float(value), 1)
        self.speed_var.set(rounded_val)
        self.save_config()
    
    def load_fonts(self):
        font_files = [
            'Montserrat-Black.ttf', 
            'Montserrat-Bold.ttf', 
            'Montserrat-ExtraBold.ttf',
            'Montserrat-Medium.ttf',  
            'Liberisco.ttf'
        ]
        
        current_os = platform.system()

        for f_name in font_files:
            font_path = FONTS / f_name
            
            if current_os == "Windows":
                ctypes.windll.gdi32.AddFontResourceExW(str(font_path), 0x10, 0)
                
            elif current_os == "Darwin":
                pyglet.font.add_file(str(font_path))
                
            elif current_os == "Linux":
                pyglet.font.add_file(str(font_path))
    
    def speed_adjust(self, input_wav, speed_multiplier):
        # vPhysically stretches audio in Cache_temp using FFmpeg
        if abs(speed_multiplier - 1.0) < 0.01:
            return input_wav 

        t_file = tempfile.NamedTemporaryFile(suffix="_stretched.wav", dir=self.specific_temp_dir, delete=False)
        output_wav = t_file.name
        t_file.close()

        # Chain atempo filters (FFmpeg limit is 0.5 to 2.0 per instance)
        if speed_multiplier > 2.0:
            passes = []
            tmp = speed_multiplier
            while tmp > 2.0:
                passes.append("atempo=2.0")
                tmp /= 2.0
            passes.append(f"atempo={tmp}")
            filter_str = ",".join(passes)
        elif speed_multiplier < 0.5:
            passes = []
            tmp = speed_multiplier
            while tmp < 0.5:
                passes.append("atempo=0.5")
                tmp /= 0.5
            passes.append(f"atempo={tmp}")
            filter_str = ",".join(passes)
        else:
            filter_str = f"atempo={speed_multiplier}"

        cmd = ["ffmpeg", "-y", "-i", input_wav, "-filter:a", filter_str, "-vn", output_wav]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_wav
        except Exception as e:
            print(f"FFmpeg Speed Error: {e}")
            return input_wav
    
    def load_selected_to_trimmer(self, event):
        try:
            index = self.audio_textbox.index(f"@{event.x},{event.y}")
            line_number = int(index.split('.')[0]) - 1
            
            if 0 <= line_number < len(self.audio_files):
                selected_file = self.audio_files[line_number]
                self.current_trim_file = selected_file
                self.tabview.set("Duration")
                self.display_waveform(selected_file)
            else:
                print(f"Clicked line {line_number}, but only {len(self.audio_files)} files loaded.")
        except Exception as e:
            print(f"Selection failed: {e}")
    
    def display_waveform(self, file_path):
        self.current_editing_path = file_path
        
        for widget in self.wave_container.winfo_children():
            widget.destroy()
        self.loading_lbl = ctk.CTkLabel(self.wave_container, text="Loading Waveform...", font=self.fontME)
        self.loading_lbl.pack(expand=True)
        threading.Thread(target=self._load_waveform_data, args=(file_path,), daemon=True).start()

    def _load_waveform_data(self, file_path):
        try:
            probe = subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', file_path
            ]).decode('utf-8').strip()
            duration = float(probe)

            target_points = 1500
            cmd = [
                'ffmpeg', '-i', file_path,
                '-ar', '1000', '-ac', '1', '-f', 's8', '-acodec', 'pcm_s8', '-'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_data, _ = process.communicate()
            y = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32)
            
            peak = np.max(np.abs(y))
            if peak > 0:
                margin = 0.8 
                y = (y / peak) * margin

            if len(y) > target_points:
                y_reshaped = y[:len(y) // target_points * target_points].reshape(target_points, -1)
                y_solid = np.max(np.abs(y_reshaped), axis=1)
            else:
                y_solid = np.abs(y)

            self.after(0, lambda: self._draw_optimized_plot(y_solid, duration, file_path))
            
        except Exception as e:
            print(f"Fast Load failed: {e}")

    def _draw_optimized_plot(self, y, duration, file_path):
        self.loading_lbl.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#1A1A1A", dpi=80)
        ax.fill_between(np.linspace(0, duration, len(y)), y, -y, color="#FF8C42", lw=0)
        
        ax.set_facecolor("#1A1A1A")
        ax.set_ylim(-1, 1)
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        if file_path in self.trim_data:
            self.trim_start, self.trim_end = self.trim_data[file_path]
        else:
            self.trim_start, self.trim_end = 0.0, duration

        # Shrouds at the correct saved positions
        self.shroud_left = ax.axvspan(0, self.trim_start, color="#FF8A423B", alpha=0.6)
        self.shroud_right = ax.axvspan(self.trim_end, duration, color="#FF8A423B", alpha=0.6)

        # Initialize Lines at the correct saved positions
        self.line_start = ax.axvline(self.trim_start, color='white', lw=2)
        self.line_end = ax.axvline(self.trim_end, color='white', lw=2)

        # Update info label immediately to reflect saved data
        self.wave_info_label.configure(
            text=f"Editing: {os.path.basename(file_path)} | Start: {self.trim_start:.2f}s | End: {self.trim_end:.2f}s"
        )

        canvas = FigureCanvasTkAgg(fig, master=self.wave_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        self.active_line = None

        def on_motion(event):
            if event.xdata is None: return
            
            near_start = abs(event.xdata - self.trim_start) < (duration * 0.03)
            near_end = abs(event.xdata - self.trim_end) < (duration * 0.03)
            canvas_widget.config(cursor="sb_h_double_arrow" if (near_start or near_end) else "")

            if self.active_line:
                if self.active_line == "start" and event.xdata < self.trim_end:
                    self.trim_start = max(0, event.xdata)
                    self.line_start.set_xdata([self.trim_start])
                    self.shroud_left.set_xy([[0, -1], [0, 1], [self.trim_start, 1], [self.trim_start, -1]])

                elif self.active_line == "end" and event.xdata > self.trim_start:
                    self.trim_end = min(duration, event.xdata)
                    self.line_end.set_xdata([self.trim_end])
                    self.shroud_right.set_xy([[self.trim_end, -1], [self.trim_end, 1], [duration, 1], [duration, -1]])
                
                canvas.draw_idle()
                self.update_idletasks()

        canvas.mpl_connect('button_press_event', lambda e: self._on_wave_press(e, duration))
        canvas.mpl_connect('motion_notify_event', on_motion)
        canvas.mpl_connect('button_release_event', self._on_wave_release)

    def _on_wave_press(self, event, duration):
        if event.xdata is None: return
        if abs(event.xdata - self.trim_start) < (duration * 0.03):
            self.active_line = "start"
        elif abs(event.xdata - self.trim_end) < (duration * 0.03):
            self.active_line = "end"

    def _on_wave_release(self, event):
        self.active_line = None
        self.wave_info_label.configure(text=f"Editing: {os.path.basename(self.current_editing_path)} | Start: {self.trim_start:.2f}s | End: {self.trim_end:.2f}s")

    def apply_trim(self):
        if hasattr(self, 'current_editing_path'):
            # Save the coordinates to our dictionary
            self.trim_data[self.current_editing_path] = (self.trim_start, self.trim_end)
            messagebox.showinfo("Trim Applied", 
                                f"Settings saved for {os.path.basename(self.current_editing_path)}\n"
                                f"Process will run from {self.trim_start:.2f}s to {self.trim_end:.2f}s")
        else:
            messagebox.showwarning("Warning", "No file is currently being edited.")

    def setup_ui(self):
        # Load fonts
        self.load_fonts()

        self.fontME = ctk.CTkFont(family="Montserrat Medium", size=12)
        self.fontBL = ctk.CTkFont(family="Montserrat Black", size=12)
        self.fontEB = ctk.CTkFont(family="Montserrat ExtraBold", size=12)
        self.fontBO = ctk.CTkFont(family="Montserrat Bold", size=12)

        self.resample = ctk.CTkFont(family="Montserrat Black", size=20)
        self.title = ctk.CTkFont(family="Montserrat Black", size=18)
        self.header = ctk.CTkFont(family="Liberisco PERSONAL USE ONLY!", size=50)


        self.main_container = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)

        self.main_container1 = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container1.pack(fill="x", expand=False, padx=5, pady=(0, 5))

        ctk.CTkLabel(self.main_container, text="PyResampler", font=self.header, text_color=["#FF8C42", "#FF6505"]).pack(pady=(5,0))
        ctk.CTkLabel(self.main_container, text=f"@cadlaxa  |  ver: {self.version}", font=self.fontME).pack(pady=(0,5))

        # File Selection Section
        self.top_frame = ctk.CTkFrame(self.main_container)
        self.top_frame.pack(fill="x", padx=10, pady=(5, 0))
        self.top_frame.grid_columnconfigure(1, weight=1) 

        ctk.CTkLabel(self.top_frame, text="Audio Files:", font=self.fontBO).grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.audio_textbox = ctk.CTkTextbox(self.top_frame, height=100, font=self.fontME)
        self.audio_textbox.grid(row=0, column=1, padx=10, pady=(10,5), sticky="ew")
        self.audio_textbox.configure(state="disabled")
        self.audio_textbox.bind("<Double-1>", self.load_selected_to_trimmer)
        self.audio_textbox.drop_target_register(DND_FILES)
        self.audio_textbox.dnd_bind('<<Drop>>', self.handle_drop)

        btn_frame = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        ctk.CTkButton(btn_frame, text="Add Files", width=120, font=self.fontBO, command=self.add_audio_files).pack()
        ctk.CTkButton(btn_frame, text="Clear", width=120, font=self.fontBO, fg_color="transparent", border_width=1, command=self.clear_audio_files).pack(pady=5)
        ctk.CTkLabel(self.top_frame, text="Resampler Dir:", font=self.fontBO).grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.resampler_entry = ctk.CTkEntry(self.top_frame, font=self.fontME, textvariable=self.resampler_dir_var)
        self.resampler_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(self.top_frame, text="Browse", font=self.fontBO, width=120, command=self.select_resampler_dir).grid(row=1, column=2, padx=10, pady=(5,10))
        
        # Tabs Section
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="x", padx=10, pady=5)
        self.single_tab = self.tabview.add("Single Resampler")
        self.batch_tab = self.tabview.add("Batch Resamplers")
        self.wave_tab = self.tabview.add("Duration")

        # Single UI
        ctk.CTkLabel(self.single_tab, text="Select Resampler:", font=self.title).pack(pady=5)
        self.resampler_combo = ctk.CTkComboBox(self.single_tab, font=self.fontME ,variable=self.resampler_var, values=self.get_resamplers(), width=400, command=self.save_resampler_config)
        self.resampler_combo.pack(pady=5)

        # Batch UI
        ctk.CTkLabel(self.batch_tab, text="Select Multiple Resamplers:", font=self.title).pack(pady=5)
        self.batch_scroll = ctk.CTkScrollableFrame(self.batch_tab, height=100)
        self.batch_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.refresh_batch_resamplers()

        # Parameters Section
        self.param_frame = ctk.CTkFrame(self.main_container)
        self.param_frame.pack(fill="x", padx=10, pady=5)
        self.param_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Title: Center aligned across all columns
        ctk.CTkLabel(self.param_frame, text="Processing Parameters", font=self.title).grid(
            row=0, column=0, columnspan=7, pady=10, sticky="ew"
        )

        for i in range(7):
            self.param_frame.grid_columnconfigure(i, weight=1)

        # Row 1: Flags and Threads
        ctk.CTkLabel(self.param_frame, text="Flags:", font=self.fontBO).grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.flags_var, width=140).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Threads:", font=self.fontBO).grid(row=1, column=2, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.threads_var, width=140).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Add this in your UI initialization section
        self.clear_cache_btn = ctk.CTkButton(
            self.param_frame, 
            text="Clear Temp Cache", 
            font=self.fontBO,
            command=self.clear_cache,
            width=140
        )
        # Adjust the row/column to fit your current layout
        self.clear_cache_btn.grid(row=1, column=5, padx=5, pady=5, sticky="we")

        # Row 2: Pitch, Follow Pitch, Speed
        ctk.CTkLabel(self.param_frame, text="Pitch:", font=self.fontBO).grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.pitch_entry = ctk.CTkEntry(self.param_frame, textvariable=self.pitch_note_var, width=140)
        self.pitch_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Speed:", font=self.fontBO).grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.speed_slider = ctk.CTkSlider(
            self.param_frame, 
            from_=-5, 
            to=5, 
            number_of_steps=100,
            variable=self.speed_var, 
            command=self.update_speed_label,
            width=100
        )
        self.speed_slider.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

        self.speed_value_label = ctk.CTkLabel(self.param_frame, textvariable=self.speed_var, font=self.fontBO)
        self.speed_value_label.grid(row=2, column=4, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(self.param_frame, text="Follow Input Pitch", font=self.fontBO, variable=self.follow_pitch_var, command=self.toggle_pitch_entry).grid(
            row=2, column=5, columnspan=2, padx=5, pady=5, sticky="w"
        )

        # Row 3: Volume, Modulation
        ctk.CTkLabel(self.param_frame, text="Vol:", font=self.fontBO).grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.volume_var, width=140).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Modulation:", font=self.fontBO).grid(row=3, column=2, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.modulation_var, width=140).grid(row=3, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(self.param_frame, text="Spectrograms", font=self.fontBO, variable=self.gen_spec_var, command=lambda: self.save_config()).grid(row=4, column=1, pady=10, sticky="we")
        ctk.CTkCheckBox(self.param_frame, text="Harvest .frq", font=self.fontBO, variable=self.generate_frq_var, command=lambda: self.save_config()).grid(row=4, column=3, pady=10, sticky="we")
        ctk.CTkCheckBox(self.param_frame, text="Only .frq", font=self.fontBO, variable=self.only_frq_var, command=lambda: self.save_config()).grid(row=4, column=5, pady=10, sticky="we")

        # Output Section
        self.out_f = ctk.CTkFrame(self.main_container)
        self.out_f.pack(fill="x", padx=10, pady=5)
        self.out_f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.out_f, text="Output Dir:", font=self.fontBO).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.output_entry = ctk.CTkEntry(self.out_f, textvariable=self.output_dir_var)
        self.output_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(self.out_f, text="Browse", font=self.fontBO, width=120, command=self.select_output_dir).grid(row=0, column=2, padx=10, pady=10)

        # Wave Editor
        self.wave_info_label = ctk.CTkLabel(self.wave_tab, text="No file selected", font=self.fontBO)
        self.wave_info_label.pack(pady=5)

        self.wave_container = ctk.CTkFrame(self.wave_tab, fg_color="#1E1E1E")
        self.wave_container.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            self.wave_container, 
            text="Double-click a file in the list to view waveform", 
            font=self.fontME,
            text_color="gray"
        ).pack(expand=True)
        
        # Buttons to save settings
        self.wave_btn_frame = ctk.CTkFrame(self.wave_tab, fg_color="transparent")
        self.wave_btn_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(self.wave_btn_frame, text="*Note: go back to the batches tab to render in batches", font=self.fontBO).pack(side="left", padx=10)
        ctk.CTkButton(self.wave_btn_frame, text="Apply Trimming", font=self.fontBO, command=self.apply_trim).pack(side="right", padx=10)

        self.progress_bar = ctk.CTkProgressBar(self.main_container1)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.set(0)

        self.start_btn = ctk.CTkButton(self.main_container1, text="START RESAMPLING", font=self.resample, height=50, command=self.run_process_thread)
        self.start_btn.pack(pady=(5, 20), padx=10, fill="x")
        
    # Logic
    def save_resampler_config(self, choice):
       self.save_config(key="default_resampler", value=choice)
              
    def select_resampler_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.resampler_dir_var.set(d)
            self.save_config()
            self.resampler_combo.configure(values=self.get_resamplers())
            self.refresh_batch_resamplers()

    def select_output_dir(self):
        d = filedialog.askdirectory()
        if d: 
            self.output_dir_var.set(d)
            self.save_config()

    def toggle_pitch_entry(self, *args):
        if not hasattr(self, 'pitch_entry'):
            return

        if self.follow_pitch_var.get():
            # DISABLED STATE
            self.pitch_entry.configure(
                state="disabled", 
                text_color="gray",
                fg_color="#333333"
            )
            self.pitch_note_var.set("Auto detects pitch")
        else:
            # NORMAL STATE
            self.pitch_entry.configure(
                state="normal", 
                text_color="white", 
                fg_color="#3b3b3b"
            )
            saved_pitch = self.config.get("pitch_note", "C4")
            self.pitch_note_var.set(saved_pitch)
        self.save_config()

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

        self.batch_scroll.grid_columnconfigure(0, weight=1)
        self.batch_scroll.grid_columnconfigure(1, weight=1)
        resamplers = sorted(self.get_resamplers())
        
        self.batch_checkboxes = {}
        for i, res in enumerate(resamplers):
            row = i // 2 
            col = i % 2

            cb = ctk.CTkCheckBox(
                self.batch_scroll, 
                text=res,
                font=self.fontME
            )
            cb.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            self.batch_checkboxes[res] = cb

    def add_audio_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")])
        if files:
            for f in files:
                if f not in self.audio_files:
                    self.audio_files.append(f)
            self.update_audio_display()
    
    def handle_drop(self, event):
        files = re.findall(r'\{(.*?)\}|(\S+)', event.data)
        dropped_files = [f[0] or f[1] for f in files]
        
        valid_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
        added_any = False

        for file_path in dropped_files:
            if file_path.lower().endswith(valid_extensions):
                if file_path not in self.audio_files:
                    self.audio_files.append(file_path)
                    added_any = True
                    print(f"File accepted: {file_path}")
            else:
                print(f"Skipped invalid file: {file_path}")

        if added_any:
            # Sync with your display function
            self.update_audio_display()
        else:
            messagebox.showwarning("No Valid Files", "No supported audio files were found in your drop.")

    def clear_audio_files(self):
        self.audio_files = []
        self.trim_data = {}
        self.current_editing_path = None
        
        self.update_audio_display()
        for widget in self.wave_container.winfo_children():
            widget.destroy()
            
        ctk.CTkLabel(
            self.wave_container, 
            text="Double-click a file in the list to view waveform", 
            font=self.fontME,
            text_color="gray"
        ).pack(expand=True)
        
        if hasattr(self, 'wave_info_label'):
            self.wave_info_label.configure(text="No file selected")

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
            fig = plt.figure(figsize=(10, 4))
            y, sr = librosa.load(audio_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            ax = fig.add_subplot(111)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), 
                                    sr=sr, x_axis='time', y_axis='mel', ax=ax)
            ax.set_title(f"Spectrogram: {os.path.basename(audio_path)}")
            fig.tight_layout()
            output_path = os.path.splitext(audio_path)[0] + "_spec.png"
            fig.savefig(output_path)
            plt.close(fig) 
        except Exception as e:
            print(f"Spectrogram Error: {e}")

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
        if self.start_btn.cget("state") == "disabled":
            return
        self.start_btn.configure(state="disabled", text="PROCESSING...")
        threading.Thread(target=self.execute_resampling, daemon=True).start()

    def execute_resampling(self):
        if not self.audio_files or not self.output_dir_var.get():
            self.after(0, lambda: messagebox.showerror("Error", "Missing files or output directory"))
            self.after(0, lambda: self.start_btn.configure(state="normal", text="START RESAMPLING"))
            return
        
        self.start_btn.configure(state="disabled", text="PROCESSING...")
        errors = []

        num_files = len(self.audio_files)
        res_count = 0
        mode = self.tabview.get()
        if mode in ["Single Resampler", "Duration"]:
            res_count = 1 if self.resampler_var.get() else 0
        else:
            res_count = sum(1 for var in self.batch_checkboxes.values() if var.get())
        
        s1_per_file = 4 + (1 if self.generate_frq_var.get() else 0)
        self.total_steps = (num_files * s1_per_file) + (num_files * res_count)
        self.steps_completed = 0
        self.progress_lock = threading.Lock()
        
        # Color codes
        BLUE, YELLOW, GREEN, RED, CYAN, RESET, BOLD = "\033[94m", "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[0m", "\033[1m"

        mode = self.tabview.get()
        res_dir = self.resampler_dir_var.get()
        resamplers = []
        if mode == "Single Resampler" or mode == "Duration":
            if self.resampler_var.get(): resamplers.append(os.path.join(res_dir, self.resampler_var.get()))
        else:
            for name, var in self.batch_checkboxes.items():
                if var.get(): resamplers.append(os.path.join(res_dir, name))

        if not resamplers:
            self.after(0, lambda: messagebox.showwarning("Warning", "No resampler selected!"))
            self.after(0, lambda: self.start_btn.configure(state="normal", text="START RESAMPLING"))
            return

        speed_val = self.speed_var.get()
        divisor = 5.0
        multiplier = 1.0 if abs(speed_val) < 0.01 else (1.0 + (speed_val / divisor) if speed_val > 0 else 1.0 / (1.0 + abs(speed_val) / divisor))

        temp_map = {}
        pitch_map = {}
        os.makedirs(self.specific_temp_dir, exist_ok=True)

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
                    self.tick_progress()
                    
                    t_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=self.specific_temp_dir, delete=False)
                    normalized_path = t_file.name
                    t_file.close()
                    
                    # Apply UTAU Standards
                    seg.set_channels(1).set_sample_width(2).set_frame_rate(44100).export(normalized_path, format="wav")
                    temp_map[audio] = normalized_path
                    self.tick_progress()
                    print(f"  {GREEN}>> Temporary UTAU-compliant WAV created.{RESET}")
                else:
                    print(f"  {GREEN}>> Already UTAU-compliant. Using original file.{RESET}")
                    temp_map[audio] = audio

                # Pitch Detection (once per file)
                detected_pitch = self.detect_pitch(temp_map[audio]) if self.follow_pitch_var.get() else self.pitch_note_var.get()
                pitch_map[audio] = detected_pitch
                self.tick_progress()
                print(f"  {BOLD}>> Pitch:{RESET} {detected_pitch}")

                processed_path = self.speed_adjust(temp_map[audio], multiplier)
                if processed_path != temp_map[audio]:
                    os.remove(temp_map[audio])
                
                temp_map[audio] = processed_path
                #print(f"{GREEN}[PREP DONE]{RESET} {os.path.basename(audio)} (Speed: {multiplier:.2f}x)")

            except Exception as e:
                print(f"{RED}[PRE-PROCESS FAILED] {file_name}: {e}{RESET}")
        
        print(f"\n{BLUE}{BOLD}=== STAGE 2: CHUNKING ==={RESET}")
        chunk_map = {}
        max_chunk_sec = 30.0
        min_chunk_ms = 200

        for audio, processed_path in temp_map.items():
            chunks = []

            if not self.only_frq_var.get():
                seg = AudioSegment.from_wav(processed_path)
                total_duration = seg.duration_seconds

                if total_duration > max_chunk_sec:
                    print(f"{YELLOW}[CHUNKING]{RESET} {os.path.basename(audio)} ({total_duration:.2f}s)")
                    for start_sec in range(0, int(total_duration), int(max_chunk_sec)):
                        end_sec = min(start_sec + max_chunk_sec, total_duration)
                        
                        # Merge tiny trailing chunks
                        if total_duration - start_sec < (min_chunk_ms / 1000) and start_sec > 0:
                            continue
                        if start_sec + (max_chunk_sec * 1.5) > total_duration:
                            end_sec = total_duration

                        chunk_seg = seg[start_sec*1000 : end_sec*1000]
                        t_chunk = tempfile.NamedTemporaryFile(suffix=f"_part{start_sec}.wav", dir=self.specific_temp_dir, delete=False)
                        chunk_path = t_chunk.name
                        t_chunk.close()
                        chunk_seg.export(chunk_path, format="wav")
                        chunks.append({"path": chunk_path, "index": len(chunks)})
                else:
                    chunks.append({"path": processed_path, "index": 0})
            else:
                print(f"{CYAN}[ONLY FRQ]{RESET} Skipping chunking for {os.path.basename(audio)}")
                chunks.append({"path": processed_path, "index": 0})
            
            chunk_map[audio] = chunks

            # Harvest FRQ (once per file)
            if self.generate_frq_var.get():
                print(f"\n{CYAN}{BOLD}[PYIN]{RESET} Analyzing segments for {os.path.basename(audio)}...")
                for c in chunks:
                    chunk_name = os.path.basename(c["path"])
                    print(f"  {CYAN}>> Generating FRQ for:{RESET} {chunk_name}")
                    
                    self.generate_harvest_frq(c["path"])
                    self.tick_progress()

        # STAGE 3: Multi-Resampler Rendering
        print(f"\n{BLUE}{BOLD}=== STAGE 3: RENDERING WITH RESAMPLERS ==={RESET}")
        tasks = []
        for audio, chunks in chunk_map.items():
            if audio not in temp_map: continue
            
            for res_path in resamplers:
                res_clean = os.path.splitext(os.path.basename(res_path))[0]
                
                for chunk in chunks:
                    base_name = os.path.splitext(os.path.basename(audio))[0]
                    
                    if self.only_frq_var.get():
                        chunk_out = os.path.join(self.output_dir_var.get(), f"{base_name}.wav")
                    else:
                        chunk_out = os.path.join(self.specific_temp_dir, f"{base_name}_{res_clean}_part{chunk['index']}.wav")
                    
                    tasks.append({
                        "res_exe": res_path,
                        "audio_in": chunk["path"],
                        "original_path": audio,
                        "audio_out": chunk_out,
                        "pitch": pitch_map[audio],
                        "multiplier": multiplier,
                        "is_chunk": True,
                        "res_name": res_clean,
                        "part_index": chunk["index"],
                        "only_frq": self.only_frq_var.get()
                    })

        if tasks:
            self.total_steps = len(tasks) * 3
            self.steps_completed = 0
            self.progress_lock = threading.Lock()
            
            try: num_threads = int(self.threads_var.get() or 1)
            except: num_threads = 1
            
            with ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
                self.tick_progress()
                futures = {executor.submit(self.run_resampler, t): t for t in tasks}

        # STAGE 4: STITCHING & FINAL CLEANUP
        if not self.only_frq_var.get():
            print(f"\n{BLUE}{BOLD}=== STAGE 4: STITCHING & CLEANUP ==={RESET}")
            for audio, chunks in chunk_map.items():
                for res_path in resamplers:
                    res_name = os.path.splitext(os.path.basename(res_path))[0]
                    base_name = os.path.splitext(os.path.basename(audio))[0]
                    
                    parts = [os.path.join(self.specific_temp_dir, f"{base_name}_{res_name}_part{i}.wav") 
                            for i in range(len(chunks))]
                    final_stitched_path = self.stitch_chunks(audio, res_name, parts)

                    if self.gen_spec_var.get() and final_stitched_path:
                        print(f"{CYAN}>> Generating Spectrogram for full output...{RESET}")
                        self.save_spectrogram(final_stitched_path)

            # 3. Cleanup logic
            print(f"{CYAN}>> Cleaning up temporary workspace...{RESET}")
            for audio, chunks in chunk_map.items():
                for c in chunks:
                    if os.path.exists(c["path"]):
                        try:
                            os.remove(c["path"])
                            print(f"  {YELLOW}Removed chunk:{RESET} {os.path.basename(c['path'])}")
                        except: pass

            for orig, temp_path in temp_map.items():
                if orig != temp_path and os.path.exists(temp_path):
                    try: 
                        os.remove(temp_path)
                        print(f"  {YELLOW}Removed base temp:{RESET} {os.path.basename(temp_path)}")
                    except: pass
        else:
            print(f"\n{GREEN}{BOLD}=== FRQ GENERATION COMPLETE ==={RESET}")
            for audio, chunks in chunk_map.items():
                for c in chunks:
                    if os.path.exists(c["path"]):
                        try:
                            os.remove(c["path"])
                            print(f"  {YELLOW}Removed chunk:{RESET} {os.path.basename(c['path'])}")
                        except: pass

            for orig, temp_path in temp_map.items():
                if orig != temp_path and os.path.exists(temp_path):
                    try: 
                        os.remove(temp_path)
                        print(f"  {YELLOW}Removed base temp:{RESET} {os.path.basename(temp_path)}")
                    except: pass

        self.after(0, lambda: self.finish_job(errors))
    
    def tick_progress(self):
        with self.progress_lock:
            self.steps_completed += 1
            progress_value = self.steps_completed / self.total_steps
            self.after(0, lambda v=progress_value: self.progress_bar.set(v))

    def run_resampler(self, t):

        # ANSI Color Codes
        YELLOW, GREEN, RED, CYAN, RESET, BOLD = "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[0m", "\033[1m"
        current_os = platform.system()
        
        try:
            if t.get("only_frq"):
                print(f"{CYAN}[ONLY FRQ MODE]{RESET} Analyzing {os.path.basename(t['audio_in'])}...")
                base_name = os.path.splitext(os.path.basename(t["original_path"]))[0]
                final_frq_path = os.path.join(self.output_dir_var.get(), f"{base_name}.wav")
                self.generate_harvest_frq(t["audio_in"], target_out=final_frq_path)
                
                print(f"{GREEN}[SUCCESS] FRQ Saved to:{RESET} {self.output_dir_var.get()}")
                self.tick_progress()
                return
            
            processed_input = t["audio_in"] 
            trim_key = t["original_path"]
            multiplier = t["multiplier"]
            
            with wave.open(processed_input, 'r') as f:
                current_duration_ms = (f.getnframes() / float(f.getframerate())) * 1000

            if t.get("is_chunk") and trim_key in self.trim_data:
                t_start_sec, t_end_sec = self.trim_data[trim_key]
                global_start_ms = (t_start_sec * 1000) / multiplier
                global_end_ms = (t_end_sec * 1000) / multiplier
                
                # Calculate the timeframe this specific chunk represents
                chunk_start_ms = (t["part_index"] * 30.0 * 1000)
                chunk_end_ms = chunk_start_ms + current_duration_ms
                local_offset_ms = max(0, global_start_ms - chunk_start_ms)
                local_end_ms = min(current_duration_ms, global_end_ms - chunk_start_ms)
                trimmed_len_ms = local_end_ms - local_offset_ms
                cutoff_ms = current_duration_ms - local_end_ms

            elif trim_key in self.trim_data:
                # Standard Full-File Logic (Non-chunked)
                t_start_sec, t_end_sec = self.trim_data[trim_key]
                local_offset_ms = (t_start_sec * 1000) / multiplier
                global_end_ms = (t_end_sec * 1000) / multiplier
                trimmed_len_ms = global_end_ms - local_offset_ms
                cutoff_ms = current_duration_ms - global_end_ms
            else:
                # No trimming data: Process the whole file
                local_offset_ms = 0
                trimmed_len_ms = current_duration_ms
                cutoff_ms = 0

            # Prevent rendering chunks that fall entirely outside the trim range
            if trimmed_len_ms <= 0:
                print(f"  {YELLOW}>> Skipping chunk {t['part_index']}: Outside trim range.{RESET}")
                return

            length = str(int(max(1, trimmed_len_ms)))
            offset = str(int(local_offset_ms))
            cutoff_val = str(-int(max(0, cutoff_ms)))
            cons = str(int(max(1, trimmed_len_ms - 50)))
            velocity = str(100)

            res_path = t["res_exe"]
            # arguements
            if "fader2" in os.path.basename(res_path).lower():
                # fader2: <in> <in2> <out> <pitch_hz> <length_ms> <ratio>
                cmd = [
                    res_path, 
                    processed_input, 
                    "null", 
                    t["audio_out"], 
                    str(float(self.note_to_hz(t["pitch"]))), 
                    length, 
                    "0.0"
                ]
            else:
                # Standard UTAU: <in> <out> <pitch> <vel> <flags> <offset> <length> <consonant> <cutoff> <vol> <mod> <tempo> <bend>
                cmd = [
                    res_path, 
                    processed_input, 
                    t["audio_out"], 
                    t["pitch"], 
                    velocity, 
                    self.flags_var.get() or "", 
                    offset, 
                    length, 
                    cons, 
                    cutoff_val, 
                    self.volume_var.get(), 
                    self.modulation_var.get(), 
                    "!120", 
                    "AA#1000#"
                ]

            if current_os != "Windows":
                if not os.access(res_path, os.X_OK):
                    os.chmod(res_path, os.stat(res_path).st_mode | 0o111)
                # Prepend wine to the command
                cmd = ["wine"] + cmd

            print(f"{YELLOW}{BOLD}[EXEC] {os.path.basename(res_path)} -> {os.path.basename(t['audio_out'])}{RESET}")

            print(f"\n{BOLD}Command Arguments:{RESET}")
            for i, arg in enumerate(cmd):
                print(f"  {CYAN}[{i}]{RESET} {arg}")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            if process.stdout:
                for line in process.stdout:
                    clean_line = line.strip()
                    if clean_line:
                        print(f"  {CYAN}> {RESET}{clean_line}")

            try:
                process.wait(timeout=120.0)
                self.tick_progress()
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"{RED}[TIMEOUT] Resampler killed after 120s{RESET}")
                raise

            if process.returncode == 0:
                print(f"{GREEN}[SUCCESS] Rendered: {os.path.basename(t['audio_out'])}{RESET}")
            else:
                print(f"{RED}[ERROR] Resampler exited with code {process.returncode}{RESET}")

            if processed_input != t["audio_in"] and os.path.exists(processed_input):
                os.remove(processed_input)
                
        except subprocess.CalledProcessError as e:
            print(f"{RED}[FAILED] Resampler returned error code {e.returncode}{RESET}")
            if e.stderr:
                print(f"{RED}Error Details: {e.stderr.strip()}{RESET}")

        except Exception as e:
            print(f"{RED}[ERROR] {os.path.basename(t['audio_out'])}: {e}{RESET}")
    
    def stitch_chunks(self, original_audio_path, resampler_name, chunk_results):
        print(f"\033[94m[STITCHING]\033[0m Joining parts for {os.path.basename(original_audio_path)}...")
        chunk_results.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', f)])
        
        combined = AudioSegment.empty()
        for chunk_file in chunk_results:
            if os.path.exists(chunk_file):
                try:
                    combined += AudioSegment.from_wav(chunk_file)
                    os.remove(chunk_file)
                except Exception as e:
                    print(f"  \033[91m[STITCH ERROR]\033[0m Could not read {os.path.basename(chunk_file)}: {e}")

        original_base = os.path.splitext(os.path.basename(original_audio_path))[0]
        final_name = f"{original_base}.wav" 
        
        mode = self.tabview.get()
        if mode in ["Single Resampler", "Duration"]:
            final_path = os.path.join(self.output_dir_var.get(), final_name)
        else:
            res_folder = os.path.join(self.output_dir_var.get(), resampler_name)
            os.makedirs(res_folder, exist_ok=True)
            final_path = os.path.join(res_folder, final_name)

        combined.export(
            final_path, 
            format="wav", 
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1"]
        )
        print(f"\033[92m[FINAL OUTPUT]\033[0m {final_path}")
        return final_path

    def finish_job(self, errors):
        self.start_btn.configure(state="normal", text="START RESAMPLING")
        self.progress_bar.set(0)

        if not errors:
            messagebox.showinfo("Success", "All files processed successfully!")
        else:
            error_msg = "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n...and {len(errors) - 5} more."
            
            messagebox.showerror("Processing Errors", f"Completed with errors:\n\n{error_msg}")

if __name__ == "__main__":
    app = UTAUResamplerGUI()
    app.mainloop()