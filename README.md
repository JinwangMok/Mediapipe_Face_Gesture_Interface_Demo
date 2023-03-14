# Face Gesture Interface(demo)

> This project targeting to develop a gesture interface using media-pipe framework for contributing to industry-academia cooperation.

## Demo
[![Demo video](https://img.youtube.com/vi/392A08BjkZM/maxresdefault.jpg)](https://youtu.be/392A08BjkZM)

## Available operations

|Operation|Action|
|:-|:-|
|Movement of the cursor|Movement of the face|
|Left click|Blink both eyes (single time)|
|Double click|Blink both eyes (two times)|
|Right click|Blink single eye (single time)|
|Drag|Move face while right-clicking|
|Drop|Open closed eye while dragging|
|Scroll-up|Roll face to the left|
|Scroll-down|Roll face to the right|
|Interface-enable/disable|Close both eyes more than 3s and then Open|

## Usage

1. Download zip file and unzip.

2. Change your directory to this project root.

3. (option)Create virtual environment using anaconda.
> ```
> conda env create -f py_face_gesture.yaml
> conda activate py_face_gesture
> ```
>
> If you are a M1 mac user, then MUST execute these codes in x86 terminal(open with rosseta).
> 
> If you are not, reference `py_face_gesture.yaml` file.

4. Please go to the root directory of this project, and then run code like this.
> ```
> python3 main.py --cam_num=0
> ```
>
> There are three setable argment `cam_num`, `canvas_width`, `canvas_height`.
>
> (canvas is a cv2.Mat type image that will show cursor movement.)