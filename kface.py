import cv2, os, yaml
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from glob import glob
from deepface import DeepFace


class Face_img:
    def __init__(self, id):
        self.id = id
        self.crop_dir = '/data/Face_id/MediaPipe/data_prep/tmp/'
        self.vec_id = np.array([int(self.id)], dtype='int64')

    def MediaPipe_face_detection(self):
        mp_face_detection = mp.solutions.face_detection

        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            self.image = cv2.imread(self.file_path)
            
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            # Crop and save face detections.
            if results.detections:
                self.crop_save_detected_bbox(results)
                

    def crop_save_detected_bbox(self, results, save=True):
        
        height, width, _ = self.image.shape
        for i, detection in enumerate(results.detections):
            # Get bounding box in absolute pixel values
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * width)
            y_min = int(bbox.ymin * height)
            box_width = int(bbox.width * width)
            box_height = int(bbox.height * height)

            # Make sure box stays within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_min + box_width)
            y_max = min(height, y_min + box_height)

            # Crop and save the face region
            self.cropped_face = self.image[y_min:y_max, x_min:x_max]

            if save:
                self.crop_path = f'{self.crop_dir}face_{i}.jpg'
                cv2.imwrite(self.crop_path, self.cropped_face)


class KFace:

    def __init__(self, cfg):
        with open(cfg, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.root_dir = self.cfg["root_dir"]
        self.face_ids = list(map(lambda x: x.split("/")[-1], glob(os.path.join(self.root_dir, "*"))))
        self.accessory = self.cfg["accessory"]
        self.cam_angle = self.cfg["cam_angle"]
        self.light = self.cfg["light"]
        self.facial_exp = self.cfg["facial_exp"]

    
    def get_image_info(self, id, acc, lux, light_vert, light_hori, face_exp, cam_vert, cam_hor) -> Face_img:

        face = Face_img(id)
        face.face_exp = face_exp
        face.acc = acc

        for light_key, light_value in self.light.items():
            if light_value == [lux, light_vert, light_hori]:
                lkey = light_key                
                face.lux = lux
                face.light_vert = light_vert
                face.light_hori = light_hori

        for pose_key, pose_value in self.cam_angle.items():
            if pose_value == [cam_vert, cam_hor]:
                pkey = pose_key
                face.cam_vert = cam_vert
                face.cam_hori = cam_hor

        img = os.path.join(self.root_dir, id, self.accessory[acc], lkey, self.facial_exp[face_exp], pkey) + ".jpg"
        face.file_path = img
        return face

    def show_image(self, face: Face_img):
        image = cv2.imread(face.file_path)

        # OpenCV는 BGR 형식으로 이미지를 읽기 때문에 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 시각화
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.title(f'ID: {face.id}\nAccessary: {face.acc}\nLux: {face.lux}\nLight: {face.light_vert},{face.light_hori}\nFacial Expression: {face.face_exp}\nCamera Angle: {face.cam_vert},{face.cam_hori}')
        plt.axis('off')  # 축 제거
        plt.show()


    def get_image_representation(self, face: Face_img, detector = "retinaface", embedder = "ArcFace", dtype = "float32") -> np.array:
        if detector == "skip":
            imgPath = face.crop_path
        else:
            imgPath = face.file_path
        target_representation = DeepFace.represent(img_path = imgPath, detector_backend= detector, model_name = embedder)[0]["embedding"]
        target_representation = np.array(target_representation, dtype= dtype)
        target_representation = np.expand_dims(target_representation, axis=0)
        face.representation = target_representation