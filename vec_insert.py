import sqlite3
import pandas as pd
import numpy as np
import faiss, os

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
            

if __name__ == "__main__":
    vecidnote = pd.read_csv("./vecid.csv")
    result2 = vecidnote.set_index("name")["uuid"].to_dict()
    personids = list(result2.keys())
    conn = sqlite3.connect("/data/Face_identity/db.sqlite3")
    cursor = conn.cursor()
    result3 = {}
    faissdb = FAISS_FlatL2(512, "/data/Face_identity/faissDB")
    faissdb_db = faissdb.load_index("faissDB.index")
    
    idx = 1000
    for personid in list(personids):
        
        cursor.execute("SELECT embedvec FROM vectorstore_vecmanager WHERE personid = ?", (personid,))
        rows = cursor.fetchall()
        for row in rows:
            embedvec = row[0]

            # Case 1: Already a list-like string (e.g., "[0.1, 0.2, ...]")
            if isinstance(embedvec, str):
                vec = np.fromstring(embedvec.strip("[]"), sep=",")
            # Case 2: Stored as BLOB (e.g., from np.array.tobytes())
            elif isinstance(embedvec, bytes):
                vec = np.frombuffer(embedvec, dtype=np.float32)
            # Case 3: Already a Python list
            elif isinstance(embedvec, (list, tuple)):
                vec = np.array(embedvec, dtype=np.float32)
            else:
                raise TypeError(f"Unexpected type {type(embedvec)} for embedvec")
            result3[personid] = vec

        faissdb.add_vec_to_index(result3[personid].reshape(1, -1) , idx)
        idx += 1
    conn.close()
    faissdb.save_index("faissDB.index")
    