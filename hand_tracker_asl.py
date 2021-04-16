import numpy as np
import copy
import itertools
import collections
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
import time
import argparse

characters = ['A', 'B', 'C', 'D', 
              'E', 'F', 'G', 'H', 
              'I', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 
              'R', 'S', 'T', 'U', 
              'V', 'W', 'X', 'Y']

FINGER_COLOR = [(128, 128, 128), (80, 190, 168), 
         (234, 187, 105), (175, 119, 212), 
         (81, 110, 221)]

JOINT_COLOR = [(0, 0, 0), (125, 255, 79), 
            (255, 102, 0), (181, 70, 255), 
            (13, 63, 255)]

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    return resized

class HandTrackerASL:
    def __init__(self,
                pd_path="models/palm_detection_6_shaves.blob", 
                pd_score_thresh=0.65, pd_nms_thresh=0.3,
                lm_path="models/hand_landmark_6_shaves.blob",
                lm_score_threshold=0.5,
                show_landmarks=True,
                show_hand_box=True,
                asl_path="models/hand_asl_6_shaves.blob",
                asl_recognition=True,
                show_asl=True):

        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.asl_path = asl_path
        self.show_landmarks=show_landmarks
        self.show_hand_box = show_hand_box
        self.asl_recognition = asl_recognition
        self.show_asl = show_asl

        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        self.preview_width = 576
        self.preview_height = 324

        self.frame_size = None

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName='HelveticaNeue.ttf', id=0)

        self.right_char_queue = collections.deque(maxlen=5)
        self.left_char_queue = collections.deque(maxlen=5)

        self.previous_right_char = ""
        self.right_sentence = ""
        self.previous_right_update_time = time.time()
        self.previous_left_char = ""
        self.left_sentence = ""
        self.previous_left_update_time = time.time()


    def create_pipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        self.pd_input_length = 128

        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        print("Creating Hand Landmark Neural Network...")          
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        self.lm_input_length = 224
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

        print("Creating Hand ASL Recognition Neural Network...")          
        asl_nn = pipeline.createNeuralNetwork()
        asl_nn.setBlobPath(str(Path(self.asl_path).resolve().absolute()))
        self.asl_input_length = 224
        asl_in = pipeline.createXLinkIn()
        asl_in.setStreamName("asl_in")
        asl_in.out.link(asl_nn.input)
        asl_out = pipeline.createXLinkOut()
        asl_out.setStreamName("asl_out")
        asl_nn.out.link(asl_out.input)

        print("Pipeline created.")
        return pipeline


    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        mpu.detections_to_rect(self.regions)
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    
    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Squeeze"))
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/self.lm_input_length)
        region.landmarks = lm


    def lm_render(self, frame, original_frame, region):
        cropped_frame = None
        hand_bbox = []
        if region.lm_score > self.lm_score_threshold:
            palmar = True
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
            if self.show_landmarks:
                list_connections = [[0, 1, 2, 3, 4], 
                                    [5, 6, 7, 8], 
                                    [9, 10, 11, 12],
                                    [13, 14 , 15, 16],
                                    [17, 18, 19, 20]]
                palm_line = [np.array([lm_xy[point] for point in [0, 5, 9, 13, 17, 0]])]

                # Draw lines connecting the palm
                if region.handedness > 0.5:
                    # Simple condition to determine if palm is palmar or dorasl based on the relative 
                    # position of thumb and pinky finger
                    if lm_xy[4][0] > lm_xy[20][0]:
                        cv2.polylines(frame, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.polylines(frame, palm_line, False, (128, 128, 128), 2, cv2.LINE_AA)
                else:
                    # Simple condition to determine if palm is palmar or dorasl based on the relative 
                    # position of thumb and pinky finger
                    if lm_xy[4][0] < lm_xy[20][0]:
                        cv2.polylines(frame, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.polylines(frame, palm_line, False, (128, 128, 128), 2, cv2.LINE_AA)
                
                # Draw line for each finger
                for i in range(len(list_connections)):
                    finger = list_connections[i]
                    line = [np.array([lm_xy[point] for point in finger])]
                    if region.handedness > 0.5:
                        if lm_xy[4][0] > lm_xy[20][0]:
                            palmar = True
                            cv2.polylines(frame, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
                            for point in finger:
                                pt = lm_xy[point]
                                cv2.circle(frame, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
                        else:
                            palmar = False
                    else:
                        if lm_xy[4][0] < lm_xy[20][0]:
                            palmar = True
                            cv2.polylines(frame, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
                            for point in finger:
                                pt = lm_xy[point]
                                cv2.circle(frame, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
                        else:
                            palmar = False
                    
                    # Use different colour for the hand to represent dorsal side
                    if not palmar:
                        cv2.polylines(frame, line, False, (128, 128, 128), 2, cv2.LINE_AA)
                        for point in finger:
                            pt = lm_xy[point]
                            cv2.circle(frame, (pt[0], pt[1]), 3, (0, 0, 0), -1)

            # Calculate the bounding box for the entire hand
            max_x = 0
            max_y = 0
            min_x = frame.shape[1]
            min_y = frame.shape[0]
            for x,y in lm_xy:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            box_width = max_x - min_x
            box_height = max_y - min_y
            x_center = min_x + box_width / 2
            y_center = min_y + box_height / 2

            # Enlarge the hand bounding box for drawing use
            draw_width = box_width/2 * 1.2
            draw_height = box_height/2 * 1.2
            draw_size = max(draw_width, draw_height)

            draw_min_x = int(x_center - draw_size)
            draw_min_y = int(y_center - draw_size)
            draw_max_x = int(x_center + draw_size)
            draw_max_y = int(y_center + draw_size)

            hand_bbox = [draw_min_x, draw_min_y, draw_max_x, draw_max_y]

            if self.show_hand_box:

                cv2.rectangle(frame, (draw_min_x, draw_min_y), (draw_max_x, draw_max_y), (36, 152, 0), 2)

                palmar_text = ""
                if region.handedness > 0.5:
                    palmar_text = "Right: "
                else:
                    palmar_text = "Left: "
                if palmar:
                    palmar_text = palmar_text + "Palmar"
                else:
                    palmar_text = palmar_text + "Dorsal"
                self.ft.putText(img=frame, text=palmar_text , org=(draw_min_x + 1, draw_max_x + 15 + 1), fontHeight=14, color=(0, 0, 0), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                self.ft.putText(img=frame, text=palmar_text , org=(draw_min_x, draw_max_x + 15), fontHeight=14, color=(255, 255, 255), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)

            if self.asl_recognition:
                # Enlarge the hand bounding box for image cropping 
                new_width = box_width/2 * 1.5
                new_height = box_height/2 * 1.5

                new_size = max(new_width, new_height)

                min_x = int(x_center - new_size)
                min_y = int(y_center - new_size)
                max_x = int(x_center + new_size)
                max_y = int(y_center + new_size)
                
                if min_x < 0:
                    min_x = 0
                if min_y < 0:
                    min_y = 0
                if max_x > frame.shape[1]:
                    max_x = frame.shape[1] - 1
                if max_y > frame.shape[0]:
                    max_y = frame.shape[0] - 1

                # Crop out the image of the hand
                cropped_frame = original_frame[min_y:max_y, min_x:max_x]

        return cropped_frame, region.handedness, hand_bbox


    def run(self):
        device = dai.Device(self.create_pipeline())
        device.startPipeline()

        q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        q_pd_in = device.getInputQueue(name="pd_in")
        q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
        q_lm_in = device.getInputQueue(name="lm_in")
        q_asl_out = device.getOutputQueue(name="asl_out", maxSize=4, blocking=True)
        q_asl_in = device.getInputQueue(name="asl_in")

        while True:
            in_video = q_video.get()
            video_frame = in_video.getCvFrame()

            h, w = video_frame.shape[:2]
            self.frame_size = max(h, w)
            self.pad_h = int((self.frame_size - h)/2)
            self.pad_w = int((self.frame_size - w)/2)

            video_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            
            frame_nn = dai.ImgFrame()
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
            q_pd_in.send(frame_nn)

            annotated_frame = video_frame.copy()

            # Get palm detection
            inference = q_pd_out.get()
            self.pd_postprocess(inference)

            # Send data for hand landmarks
            for i,r in enumerate(self.regions):
                img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                q_lm_in.send(nn_data)

            # Retrieve hand landmarks
            for i,r in enumerate(self.regions):
                inference = q_lm_out.get()
                self.lm_postprocess(r, inference)
                hand_frame, handedness, hand_bbox = self.lm_render(video_frame, annotated_frame, r)
                # ASL recognition
                if hand_frame is not None and self.asl_recognition:
                    hand_frame = cv2.resize(hand_frame, (self.asl_input_length, self.asl_input_length), interpolation=cv2.INTER_NEAREST)
                    hand_frame = hand_frame.transpose(2,0,1)
                    nn_data = dai.NNData()
                    nn_data.setLayer("input", hand_frame)
                    q_asl_in.send(nn_data)
                    asl_result = np.array(q_asl_out.get().getFirstLayerFp16())
                    asl_idx = np.argmax(asl_result)
                    # Recognized ASL character is associated with a probability
                    asl_char = [characters[asl_idx], round(asl_result[asl_idx] * 100, 1)]
                    selected_char = asl_char
                    current_char_queue = None
                    if handedness > 0.5:
                        current_char_queue = self.right_char_queue
                    else:
                        current_char_queue = self.left_char_queue
                    current_char_queue.append(selected_char)
                    # Peform filtering of recognition resuls using the previous 5 results
                    # If there aren't enough reults, take the first result as output
                    if len(current_char_queue) < 5:
                        selected_char = current_char_queue[0]
                    else:
                        char_candidate = {}
                        for i in range(5):
                            if current_char_queue[i][0] not in char_candidate:
                                char_candidate[current_char_queue[i][0]] = [1, current_char_queue[i][1]]
                            else:
                                char_candidate[current_char_queue[i][0]][0] += 1
                                char_candidate[current_char_queue[i][0]][1] += current_char_queue[i][1]
                        most_voted_char = ""
                        max_votes = 0
                        most_voted_char_prob = 0
                        for key in char_candidate:
                            if char_candidate[key][0] > max_votes:
                                max_votes = char_candidate[key][0]
                                most_voted_char = key
                                most_voted_char_prob = round(char_candidate[key][1] / char_candidate[key][0], 1)
                        selected_char = (most_voted_char, most_voted_char_prob)
                    
                    if self.show_asl:
                        gesture_string = "Letter: " + selected_char[0] + ", " + str(selected_char[1]) + "%"
                        textSize = self.ft.getTextSize(gesture_string, fontHeight=14, thickness=-1)[0]
                        cv2.rectangle(video_frame, (hand_bbox[0] - 5, hand_bbox[1]), (hand_bbox[0] + textSize[0] + 5, hand_bbox[1] - 18), (36, 152, 0), -1)
                        self.ft.putText(img=video_frame, text=gesture_string , org=(hand_bbox[0], hand_bbox[1] - 5), fontHeight=14, color=(255, 255, 255), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)

            video_frame = video_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]
            cv2.imshow("hand tracker", video_frame)
            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('1'):
                self.show_hand_box = not self.show_hand_box
            elif key == ord('2'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('3'):
                self.show_asl = not self.show_asl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd_m", default="models/palm_detection_6_shaves.blob", type=str,
                        help="Path to a blob file for palm detection model (default=%(default)s)")
    parser.add_argument("--lm_m", default="models/hand_landmark_6_shaves.blob", type=str,
                        help="Path to a blob file for landmark model (default=%(default)s)")
    parser.add_argument("--asl_m", default="models/hand_asl_6_shaves.blob", type=str,
                        help="Path to a blob file for ASL recognition model (default=%(default)s)")
    parser.add_argument('-asl', '--asl', default=True, 
                        help="enable ASL recognition")
    args = parser.parse_args()

    ht = HandTrackerASL(pd_path=args.pd_m, lm_path=args.lm_m, asl_path=args.asl_m, asl_recognition=args.asl)
    ht.run()
