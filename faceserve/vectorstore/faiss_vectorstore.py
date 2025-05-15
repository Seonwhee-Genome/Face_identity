import faiss, os
import numpy as np


class FAISS_FlatL2:    

    def __init__(self, dim, root_dir="/data/Face_identity/"):        
        self.dim = dim
        self.root = root_dir

    def create_index(self):
        self.indexFlat = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(self.indexFlat)  
        self.all_ids = faiss.vector_to_array(self.index.id_map)

    def add_vec_to_index(self, vec, id):
        target_id = int(id)
        
        if target_id in list(self.all_ids):
            # FAISS expects IDs in a `faiss.IDSelector` object
            selector = faiss.IDSelectorBatch(np.array([target_id], dtype=np.int64))
            # Remove the ID
            self.index.remove_ids(selector)
            self.index.add_with_ids(vec, id)            
        else:            
            self.index.add_with_ids(vec, id)
        self.all_ids = faiss.vector_to_array(self.index.id_map)
            

    def save_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        faiss.write_index(self.index, self.index_path)


    def load_index(self, index_file_name):
        self.index_path = os.path.join(self.root, index_file_name)
        self.index = faiss.read_index(self.index_path)
        self.all_ids = faiss.vector_to_array(self.index.id_map)


    def search_index(self, vec, topk=2, threshold = 5.0):
        try:
            print(vec)
            Distance, ID = self.index.search(vec, topk)
            print(Distance)
            return self.format_faiss_results(Distance, ID, threshold)

        except ValueError as ve:
            print(ve)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL", 'http_error':500}            
        except AssertionError as ae:
            print(ae)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL", 'http_error':500}
        except SyntaxError as se:
            print(se)
            return {'message': 'AI가 임베딩한 벡터의 형식이 잘못되었습니다.', 'status': "FAIL", 'http_error':500}
        

    def format_faiss_results(self, D, I, threshold = 5.0):
        result = {}        
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            result[f"top {rank} id"] = int(idx)
            result[f"top {rank} distance"] = float(dist)

        ## Only takes the Top-1 distance into account 
        if D[0][0] < threshold:            
            result['status'] = "IDENTIFIED"
        else:
            result['status'] = "UNIDENTIFIED"
            
        return result


    def delete_vec_from_index(self, id):
        # Get the internal index for a specific ID (e.g., ID 1003)
        target_id = int(id)        
        if target_id in list(self.all_ids):
            # FAISS expects IDs in a `faiss.IDSelector` object
            selector = faiss.IDSelectorBatch(np.array([target_id], dtype=np.int64))
            # Remove the ID
            self.index.remove_ids(selector)            
            self.all_ids = faiss.vector_to_array(self.index.id_map)            
            return 1
        else:
            return 0
        