import faiss, os
from kface import KFace, Face_img


class FAISS_FlatL2:    

    def __init__(self, dim, root_dir="/data/Face_identity/"):        
        self.dim = dim
        self.root = root_dir

    def create_index(self):
        self.indexFlat = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(self.indexFlat)  
        

    def add_vec_to_index(self, obj: Face_img):
        self.index.add_with_ids(obj.representation, obj.vec_id)


    def save_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        faiss.write_index(index, self.index_path)


    def load_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        self.index = faiss.read_index(self.index_path)


    def search_index(self, obj: Face_img, topk=2):
        Distance, ID = self.index.search(obj.representation, topk)        
        return self.format_faiss_results(Distance, ID)
        

    def format_faiss_results(self, D, I):
        result = {}
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            result[f"top {rank}"] = int(idx)
            result[f"top {rank} distance"] = float(dist)
        return result


if __name__=="__main__":
    kface = KFace("/data/Face_id/DeepFace/data_prep/cfg_kface.yaml")
    face1 = kface.get_image_info(kface.face_ids[7], "No_acc", 400, 180, 180, "poker", 0, 0)
    face1.MediaPipe_face_detection()
    kface.get_image_representation(face1, detector="skip", embedder="Facenet512")

    face2 = kface.get_image_info(kface.face_ids[8], "No_acc", 400, 180, 180, "poker", 0, 0)
    face2.MediaPipe_face_detection()
    kface.get_image_representation(face2, detector="skip", embedder="Facenet512")

    face3 = kface.get_image_info(kface.face_ids[9], "No_acc", 400, 180, 180, "poker", 0, 0)
    face3.MediaPipe_face_detection()
    kface.get_image_representation(face3, detector="skip", embedder="Facenet512")

    face4 = kface.get_image_info(kface.face_ids[10], "No_acc", 400, 180, 180, "poker", 0, 0)
    face4.MediaPipe_face_detection()
    kface.get_image_representation(face4, detector="skip", embedder="Facenet512")

    vectorstore = FAISS_FlatL2(512)
    vectorstore.create_index()
    vectorstore.add_vec_to_index(face1)
    vectorstore.add_vec_to_index(face2)
    vectorstore.add_vec_to_index(face3)
    vectorstore.add_vec_to_index(face4)

    print(vectorstore.search_index(face1))
    print(vectorstore.search_index(face2))