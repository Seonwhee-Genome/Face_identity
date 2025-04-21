import faiss, os


class FAISS_FlatL2:    

    def __init__(self, dim, root_dir="/data/Face_identity/"):        
        self.dim = dim
        self.root = root_dir

    def create_index(self):
        self.indexFlat = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(self.indexFlat)  
        

    def add_vec_to_index(self, vec, id):
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