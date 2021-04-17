# American Sign Language (ASL) Recognition using Hand Landmarks

This is the source code of the article: 


## Install dependencies

On a Raspberry Pi 4B, run in terminal:

```
git clone https://github.com/cortictechnology/hand_asl_recognition.git
cd hand_asl_recognition
bash install_dependencies.sh
```

## To run

1. Make sure the OAK-D device is plug into the Pi.
2. In the terminal, run

```
python3 hand_tracker_asl.py
```

By default, ASL recognition is enabled.


## Model description

In the models folder, 3 models are provided:

1. palm_detection_6_shaves.blob: This is the palm detection model. Converted using OpenVino's myriad_compiler.
2. hand_landmark_6_shaves.blob: This is the model to detect the hand landmarks using the palm detection model. Converted using OpenVino's myriad_compiler.
3. hand_asl_6_shaves.blob: This is the model to classify the hand's gesture into ASL characters. Converted using OpenVino's myriad_compiler.

## To train your own ASL recognition (or any gesture classification model)

Please refer to the training script in the training folder. We have provided all of the data we used for training the ASL recognition model.
You can change the data or modify the training script to train your own model. The training script will save the trained model into a frozen PB model, 
which can then be converted to run on the OAK-D hardware using OpenVino's mo.py scrip and myriad_compiler.


## Credits
* [Google Mediapipe](https://github.com/google/mediapipe)
* [depthai_hand_tracker](https://github.com/geaxgx/depthai_hand_tracker)
* [Pinto](https://github.com/PINTO0309) for the model convertion tools.
* [tello-gesture-control](https://github.com/kinivi/tello-gesture-control) for his pose history model.
* [David Lee](https://github.com/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5)
* [David's dataset on Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters)
