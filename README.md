# PyResampler
Python GUI toolkit for passing down Resampler arguments (single and batch resampling)

---
## To do:
- [x] UI improvements
- [x] Control output speed
- [x] Use wine for resamplers (for mac and linux)
- [ ] Switch to pure python based FFpmeg and FFprobe (so mac and linux can use them too)
- [x] Use custom fonts
- [x] Multi-threaded resampling
- [x] Option to generate frq files with Harvest
- [x] Accept any type of audio files
- [x] Convert to wav (temp file) before sending it to the resampler
- [x] Fader2.exe support
- [x] Audio length/duration editor
- [x] Render long audios in chunks
---
## How to use:
- Install necessary modules in order for the GUI script to run.
    ```
    pip install -r requirements.txt
    ```
- The GUI will save the resampler and output directory in the config.yaml file.
- You also need to download `ffmpeg`, `ffprobe`, and `ffplay` in order to for this toolkit to work
    - https://www.ffmpeg.org/download.html#build
- After downloading, place the files in **`Assets/Ffmpeg/(your os [win, macos, linux])`**
- For macos and linux devices, you need to have Wine binaries so you can run the exe resamplers
---
<img width="702" height="632" alt="image" src="https://github.com/user-attachments/assets/2ce09411-5a7c-4f8b-b02e-929d8dc5816f" />
