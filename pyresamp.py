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
ffmpeg_folder = os.path.join(base_path, "Ffmpeg")

os.environ["PATH"] = ffmpeg_folder + os.pathsep + os.environ["PATH"]
from pydub import AudioSegment

print("FFMPEG folder injected:", ffmpeg_folder)

yaml = YAML()
yaml.preserve_quotes = True

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("Assets/Theme/orange.json")

ASSETS = P('./Assets')
FONTS = P(ASSETS, 'Fonts')

class UTAUResamplerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        icon_path = P(__file__).parent / "Assets" / "icon.ico"
        # Set the icon
        self.iconbitmap(str(icon_path))

        self.title("PyResampler - Batch Resampling GUI")
        self.version = "0.0.5"
        self.config_path = "config.yaml"
        self.config = self.load_config()
        self.audio_files = []
        self.batch_checkboxes = {}
        self.geometry("700x600")
        self.specific_temp_dir = os.path.join(base_path, "Cache_temp")
        print(f"PyResampler Verion: {self.version}")
        self.trim_data = {}

        # Variables 
        self.resampler_dir_var = ctk.StringVar(value=self.config.get("resampler_directory", ""))
        self.output_dir_var = ctk.StringVar(value=self.config.get("output_directory", ""))
        self.resampler_var = ctk.StringVar(value=self.config.get("default_resampler", ""))
        
        self.flags_var = ctk.StringVar(value="")
        self.pitch_note_var = ctk.StringVar(value="C4")
        self.volume_var = ctk.StringVar(value="100")
        self.modulation_var = ctk.StringVar(value="100")
        self.threads_var = ctk.StringVar(value="4")

        self.setup_ui()
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
                    return yaml.load(f) or {}
            except: pass
        return {"resampler_directory": "", "default_resampler": "", "output_directory": ""}

    def update_config_file(self, key, value):
        self.config[key] = value
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f)
        except PermissionError:
            print("\033[91m[ERROR] Could not save config.yaml. Is it open in another program?\033[0m")
    
    def update_speed_label(self, value):
        rounded_val = round(float(value), 1)
        self.speed_var.set(rounded_val)
    
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
            target_sr = 1000
            y, sr = librosa.load(file_path, sr=target_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak 

            resolution = 1500
            chunk_size = len(y) // resolution
            
            if chunk_size > 0:
                y_reshaped = y[:resolution * chunk_size].reshape(resolution, chunk_size)
                y_solid = np.max(np.abs(y_reshaped), axis=1)
            else:
                y_solid = np.abs(y)

            self.after(0, lambda: self._draw_optimized_plot(y_solid, duration, file_path))
        except Exception as e:
            print(f"Load failed: {e}")

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

        self.speed_var = ctk.DoubleVar(value=0.0)
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

        self.follow_pitch_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.param_frame, text="Follow Input Pitch", font=self.fontBO, variable=self.follow_pitch_var, command=self.toggle_pitch_entry).grid(
            row=2, column=5, columnspan=2, padx=5, pady=5, sticky="w"
        )

        # Row 3: Volume, Modulation
        ctk.CTkLabel(self.param_frame, text="Vol:", font=self.fontBO).grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.volume_var, width=140).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.param_frame, text="Modulation:", font=self.fontBO).grid(row=3, column=2, padx=5, pady=5, sticky="e")
        ctk.CTkEntry(self.param_frame, textvariable=self.modulation_var, width=140).grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # Row 4: Extra Options (Checkboxes)
        self.gen_spec_var = ctk.BooleanVar(value=False)
        self.generate_frq_var = ctk.BooleanVar(value=False)
        self.only_frq_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(self.param_frame, text="Spectrograms", font=self.fontBO, variable=self.gen_spec_var).grid(row=4, column=1, pady=10, sticky="we")
        ctk.CTkCheckBox(self.param_frame, text="Harvest .frq", font=self.fontBO, variable=self.generate_frq_var).grid(row=4, column=3, pady=10, sticky="we")
        ctk.CTkCheckBox(self.param_frame, text="Only .frq", font=self.fontBO, variable=self.only_frq_var).grid(row=4, column=5, pady=10, sticky="we")

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
            self.pitch_note_var.set("C4")

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

                # Harvest FRQ (once per file)
                if self.generate_frq_var.get():
                    self.tick_progress()
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
                self.tick_progress()
                
                # Check for Only FRQ mode
                if self.only_frq_var.get():
                    self.generate_harvest_frq(temp_map[audio], target_out=out_file)
                    continue

                if not self.trim_data:
                    print(f"{YELLOW}No trim data found. All files will process at full length.{RESET}")
                else:
                    for path, coords in self.trim_data.items():
                        print(f"{GREEN}File:{RESET} {os.path.basename(path)}")
                        print(f"  {BOLD}Start:{RESET} {coords[0]:.2f}s | {BOLD}End:{RESET} {coords[1]:.2f}s")

                tasks.append({
                    "res_exe": res_path,
                    "audio_in": temp_map[audio],
                    "original_path": audio,
                    "audio_out": out_file,
                    "pitch": pitch_map[audio],
                    "multiplier": multiplier
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

        # STAGE 3: Final Cleanup
        print(f"\n{BLUE}{BOLD}=== STAGE 3: CLEANUP ==={RESET}")
        for orig, temp in temp_map.items():
            if orig != temp and os.path.exists(temp):
                try: 
                    os.remove(temp)
                    print(f"Removed temp: {os.path.basename(temp)}")
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
        
        try:
            processed_input = t["audio_in"] 
            trim_key = t["original_path"]
            multiplier = t["multiplier"]
            
            with wave.open(processed_input, 'r') as f:
                full_duration_ms = (f.getnframes() / float(f.getframerate())) * 1000

            if trim_key in self.trim_data:
                t_start_sec, t_end_sec = self.trim_data[trim_key]
                offset_ms = (t_start_sec * 1000) / multiplier
                end_ms = (t_end_sec * 1000) / multiplier
                trimmed_len_ms = end_ms - offset_ms
                cutoff_ms = full_duration_ms - end_ms
            else:
                offset_ms = 0
                trimmed_len_ms = full_duration_ms
                cutoff_ms = 0

            length = str(int(max(1, trimmed_len_ms)))
            offset = str(int(offset_ms))
            cutoff_val = str(-int(max(0, cutoff_ms)))
            cons = str(int(max(1, trimmed_len_ms - 50)))
            velocity = str(100)

            # arguements
            if "fader2" in os.path.basename(t["res_exe"]).lower():
                # fader2: <in> <in2> <out> <pitch_hz> <length_ms> <ratio>
                cmd = [
                    t["res_exe"], 
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
                    t["res_exe"], 
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

            print(f"{YELLOW}{BOLD}[EXEC] {os.path.basename(t['res_exe'])} -> {os.path.basename(t['audio_out'])}{RESET}")

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