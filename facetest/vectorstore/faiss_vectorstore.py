import faiss, os
import numpy as np
from .models import Vecmanager

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
            Distance, ID = self.index.search(vec, topk)
            print(ID)
            print(Distance)
            res, code = self.format_faiss_results(Distance, ID, threshold)
            return res, code
            

        except ValueError as ve:
            print(ve)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL"}, 400       
        except AssertionError as ae:
            print(ae)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL"}, 400
        except SyntaxError as se:
            print(se)
            return {'message': 'AI가 임베딩한 벡터의 형식이 잘못되었습니다.', 'status': "FAIL"}, 400
        

    def format_faiss_results(self, D, I, threshold = 5.0):
        result = {}
        try:
            for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
                entry = Vecmanager.objects.get(vectorid=idx)            
                result[f"top {rank} id"] = entry.personid #int(idx)
                result[f"top {rank} distance"] = float(dist)

            ## Only takes the Top-1 distance into account 
            if D[0][0] < threshold:            
                result['status'] = "IDENTIFIED"
            else:
                result['status'] = "UNIDENTIFIED"

            return result, 200
        except Vecmanager.DoesNotExist as e:
            print(e)
            return {'message' : '존재하지 않는 데이터베이스상에서 안면 정보를 찾을 수 없습니다. 지자체 user가 등록되어 있는지 확인해주세요', 'status': "FAIL"}, 404


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



class FAISS_InnerProd(FAISS_FlatL2):

    def __init__(self, dim, root_dir="/data/Face_identity/"):
        super().__init__(dim, root_dir)


    def create_index(self):
        self.indexFlat = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.indexFlat)  
        self.all_ids = faiss.vector_to_array(self.index.id_map)


    def add_vec_to_index(self, vec, id):
        target_id = int(id)
        faiss.normalize_L2(vec)
        
        if target_id in list(self.all_ids):
            # FAISS expects IDs in a `faiss.IDSelector` object
            selector = faiss.IDSelectorBatch(np.array([target_id], dtype=np.int64))
            # Remove the ID
            self.index.remove_ids(selector)
            self.index.add_with_ids(vec, id)            
        else:            
            self.index.add_with_ids(vec, id)
        self.all_ids = faiss.vector_to_array(self.index.id_map)


    def search_index(self, vec, topk=2, threshold = 5.0):
        faiss.normalize_L2(vec)
        
        try:
            Distance, ID = self.index.search(vec, topk)
            res, code = self.format_faiss_results(Distance, ID, threshold)
            return res, code
            

        except ValueError as ve:
            print(ve)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL"}, 400       
        except AssertionError as ae:
            print(ae)
            return {'message': 'AI가 임베딩한 벡터의 차원이 잘못되었습니다.', 'status': "FAIL"}, 400
        except SyntaxError as se:
            print(se)
            return {'message': 'AI가 임베딩한 벡터의 형식이 잘못되었습니다.', 'status': "FAIL"}, 400


    def format_faiss_results(self, D, I, threshold = 5.0):
        result = {}
        try:
            for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
                entry = Vecmanager.objects.get(vectorid=idx)            
                result[f"top {rank} id"] = entry.personid #int(idx)
                result[f"top {rank} distance"] = abs(1-float(dist))

            ## Only takes the Top-1 distance into account 
            if abs(1 - D[0][0]) < threshold:            
                result['status'] = "IDENTIFIED"
            else:
                result['status'] = "UNIDENTIFIED"

            return result, 200
            
        except Vecmanager.DoesNotExist as e:
            print(e)
            return {'message' : '존재하지 않는 데이터베이스상에서 안면 정보를 찾을 수 없습니다. 지자체 user가 등록되어 있는지 확인해주세요', 'status': "FAIL"}, 404
        
        

    
        
    