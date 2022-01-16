# Real Time Background Subtraction  

Official demo repository of real-time background matting of a person in foreground, as a result of diploma thesis.

![Conference call](https://github.com/AmadejTratnik/background_subtractor/blob/main/images/conference_call1.gif)

Our model can capture live webcam stream or video and produce background of a person in real time up to 30FPS, without the additional GPU or pre-captured background image.

Model architecture is based on FAST-SCNN network and optimized with ONNX Runtime accelerator.

## Installation
Download the project from Github and change your current directory:
```
$ (base) cd background_subtractor
```
Use a virtual environment to isolate your environment, and install the required dependencies.
```
$ (base) python3 -m venv venv
$ (base) source venv/bin/activate
$ (venv) pip3 install -r requirements.txt
```

## Background Subtraction
To start real time background subtraction, simply write:
```
$ (venv) python3 scripts/bacground_subtraction.py
```

By default, the script will run for 1000 frames, showing original picture, produced binary mask and final product of person's background subtraction in real time.

## Some more examples:

![Conference call 2](https://github.com/AmadejTratnik/background_subtractor/blob/main/images/conference_call2.gif)

![Close up filming](https://github.com/AmadejTratnik/background_subtractor/blob/main/images/close_up.gif)

![Dynamic camera](https://github.com/AmadejTratnik/background_subtractor/blob/main/images/dynamic_background.gif)