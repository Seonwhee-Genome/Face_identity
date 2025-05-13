import faiss, os
import numpy as np


class FAISS_FlatL2:    

    def __init__(self, dim, root_dir="/data/Face_identity/"):        
        self.dim = dim
        self.root = root_dir

    def create_index(self):
        self.indexFlat = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(self.indexFlat)  
        

    def add_vec_to_index(self, vec, id):
        target_id = int(id)
        all_ids = faiss.vector_to_array(self.index.id_map)
        if target_id in list(all_ids):
            # FAISS expects IDs in a `faiss.IDSelector` object
            selector = faiss.IDSelectorBatch(np.array([target_id], dtype=np.int64))
            # Remove the ID
            self.index.remove_ids(selector)
            self.index.add_with_ids(vec, id)            
        else:            
            self.index.add_with_ids(vec, id)
            

    def save_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        faiss.write_index(self.index, self.index_path)


    def load_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        self.index = faiss.read_index(self.index_path)


    def search_index(self, vec, topk=2):
        print(vec)
        Distance, ID = self.index.search(vec, topk)
        print(Distance)
        return self.format_faiss_results(Distance, ID)
        

    def format_faiss_results(self, D, I):
        result = {}
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            result[f"top {rank} id"] = int(idx)
            result[f"top {rank} distance"] = float(dist)
        return result


    def delete_vec_from_index(self, id):
        # Get the internal index for a specific ID (e.g., ID 1003)
        target_id = int(id)
        all_ids = faiss.vector_to_array(self.index.id_map)
        if target_id in list(all_ids):
            # FAISS expects IDs in a `faiss.IDSelector` object
            selector = faiss.IDSelectorBatch(np.array([target_id], dtype=np.int64))
            # Remove the ID
            self.index.remove_ids(selector)
            return {'message': f"사용자 {target_id}의 정보가 삭제되었습니다", 'status': "SUCCESS"}
        else:
            return {'message': f"삭제할 사용자 {target_id}의 정보가 존재하지 않습니다", 'status': "FAIL"}
        