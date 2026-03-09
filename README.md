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
- You also need to download `ffmpeg`, `ffprobe`, and `ffplay` in order to for this toolkit to work (windows build is already included in the assets folder)
    - https://www.ffmpeg.org/download.html#build
- After downloading, place the files in **`Assets/Ffmpeg/(your os [win, macos, linux])`**
- For macos and linux devices, you need to have Wine binaries so you can run the exe resamplers
---
## Main UI
<img width="410" height="380" alt="image" src="https://github.com/user-attachments/assets/9440ac1b-a1b7-413f-ab0d-b74cbabb8861" />
<img width="410" height="380" alt="image" src="https://github.com/user-attachments/assets/71924516-1e2d-4563-96a5-dee4e61dee5d" />

## Audio input & Resampler directory
- You can input audio via drag and drop or via `Add files` button
<img width="665" height="175" alt="image" src="https://github.com/user-attachments/assets/2bbcc776-fde5-4ef3-baaf-b54f2de66ea3" />

## UI Tabs
- Single Resampler picker - single resampler rendering
<img width="663" height="209" alt="image" src="https://github.com/user-attachments/assets/14f59a52-8a64-4e1a-8910-53fdc65f4e9c" />

- Batch Resampler picker - multiple resampler rendering
<img width="658" height="330" alt="image" src="https://github.com/user-attachments/assets/88c23ef6-3312-492e-b33f-35d31dc9e7df" />

- Audio trimmer/duration editor
  - `Left/Right lines`: Similar to left blank and cutoff, you can drag them to cut the audio and only render the selected part
  - `Red line`: Playhead, click the waveform to play on that exact position
<img width="666" height="340" alt="image" src="https://github.com/user-attachments/assets/3e63582e-b5c2-4b8e-949b-34740d9b38b9" />

- Settings
  - `Themes`: light and dark theme
  - `Wine path`: click the `Detect` button to automatically find the wine path (usable for macos and linux)
  - `Open button`: if the wine path detector fails, you can click this to manually find the wine path banaries
  - `Open Output`: Opens the output folder after rendering the input files
  <img width="663" height="259" alt="image" src="https://github.com/user-attachments/assets/2613a484-e466-4922-bc99-b189bd8eb6f3" />

## Processing Parameters
- `Flags`: Global resampler flag input, (applies the flags to all input files, flags varies for each resampler)
- `Threads`: Resampler thread and Harvest frq thread, the more threads to set, the faster the execution of the processes
- `Clear Temp Cache`: Clears the temp files inside the cache folder
- `Pitch`: The base pitch of the output render, you can input herts or the musical notation of the pitch (440, 390, 20, C4, G#3, D5)
- `Speed`: Speed of the rendered audio (slow or fast)
- `Follow input pitch`: Detects the average pitch of the input audio/s so you don't have to guess them, you can do the other method by making the pitch entry blank so the resampler skips the pitch input (*note: some resamplers will error or the output is unexpected from the result if the pitch entry is blank)
- `Volume`: Volume of the rendered audio
- `Modulation`: Modulation parameter of resamplers, uses the detected f0 of the audio so it copies the actual pitch of the audio (0 = flat, 100 = actual pitch, 200 = exaggerated, -100 = inverted pitch)
- `Spectrogram`: Generates the image spectrogram of the rendered audio
- `Harvest.frq`: Uses Harvest algorithm to generate the frq files instead of the resamplers (very fast and accurate, very useable if the resampler also uses frq files for the f0)
- `Only frq`: Outputs only the frq files (uses harvest)
<img width="673" height="309" alt="image" src="https://github.com/user-attachments/assets/6d375047-5e7d-4943-9ade-6c9d86409dc0" />









