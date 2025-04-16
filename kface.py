import cv2, os
import matplotlib.pyplot as plt
from glob import glob
from deepface import DeepFace
import numpy as np
import yaml

class Face_img:
    def __init__(self, id):
        self.id = id  


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