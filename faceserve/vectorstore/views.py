from django.shortcuts import render

# Create your views here.
from .serializers import VecSerializer, SearchSerializer
from .models import Vecmanager, Searchmanager
import os, requests, shutil
import datetime, re
import numpy as np
from ast import literal_eval
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.renderers import JSONRenderer
from django.shortcuts import get_object_or_404
from django.http import Http404,HttpResponse,HttpResponseRedirect, JsonResponse

from rest_framework import viewsets
from rest_framework.decorators import action
from django.db.models import Max
from uuid import UUID
from .faiss_vectorstore import FAISS_FlatL2

def FAISS_server_start(username: str):
    vstore = FAISS_FlatL2(512)
    if os.path.exists(os.path.join(vstore.root, f'faissDB-{username}.index')):
        vstore.load_index(f'faissDB-{username}.index')        
    else:
        vstore.create_index()
    print(f'지자체 {username}의 FAISS DB load')
    return vstore, f'faissDB-{username}.index'    


class RegisterViewSet(viewsets.ModelViewSet):
    queryset = Vecmanager.objects.all()
    serializer_class = VecSerializer
    

    def create(self, request, *args, **kwargs):
        request.data._mutable = True

        username = request.POST['user']
        vstore, FAISS_outfile = FAISS_server_start(username)
        representation = request.POST['embedvec']
        uuid = request.POST['personid']
        represent_list = literal_eval(representation)
        request.data['embedvec'] = represent_list

        # 🔸 Generate next available vectorid
        max_vectorid = Vecmanager.objects.aggregate(Max('vectorid'))['vectorid__max']
        next_vectorid = 1 if max_vectorid is None else max_vectorid + 1
        request.data['vectorid'] = next_vectorid

        request.data._mutable = False

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=username)
            
        # 🔸 Add to vector index
        vector = np.array(represent_list, dtype=np.float32).reshape(1, -1)
        vstore.add_vec_to_index(vector, int(next_vectorid))
        vstore.save_index(FAISS_outfile)

        return JsonResponse({'message': f"사용자 {uuid}의 정보가 등록되었습니다", 'status': "SUCCESS"})
        

    @action(detail=False, methods=['get'], url_path='dump-db')
    def dump_database(self, request):
            
        """
        Returns a list of all records in Vecmanager table.
        """
        all_entries = Vecmanager.objects.all().order_by('created_at')
        data = []
        
        for entry in all_entries:
            vstore, _ = FAISS_server_start(entry.user)
            data.append({
                'personid': str(entry.personid),
                'vectorid': entry.vectorid,
                'user': entry.user,
                'created_at': entry.created_at.isoformat(),
                'current vectorids': vstore.all_ids
            })
            del(vstore)
            
        return Response({'count': len(data), 'results': data})
        
          
    @action(detail=False, methods=['put'], url_path='update-embedvec/(?P<uuid_str>[0-9a-f-]+)')
    def update_embedvec(self, request, uuid_str=None):
        try:
            uuid_obj = UUID(uuid_str, version=4)
            record = Vecmanager.objects.get(personid=uuid_obj)

            vstore, FAISS_outfile = FAISS_server_start(record.user)

            # Get and parse new embedvec
            embedvec_input = request.data.get("embedvec", None)
            if embedvec_input is None:
                return Response({"message": "embedvec 값이 필요합니다.", "status": "FAIL"},
                                status=status.HTTP_400_BAD_REQUEST)

            # Convert string to list if needed
            if isinstance(embedvec_input, str):
                try:
                    embedvec = literal_eval(embedvec_input)
                except Exception:
                    return Response({"message": "embedvec 문자열을 리스트로 변환할 수 없습니다.", "status": "FAIL"},
                                    status=status.HTTP_400_BAD_REQUEST)
            else:
                embedvec = embedvec_input

            if not isinstance(embedvec, list) or len(embedvec) != 512:
                return Response({"message": "embedvec은 512차원의 리스트여야 합니다.", "status": "FAIL"},
                                status=status.HTTP_400_BAD_REQUEST)

            # Update record
            record.embedvec = embedvec
            record.save()

            # Update FAISS index
            vector = np.array(embedvec, dtype=np.float32).reshape(1, -1)            
            vstore.add_vec_to_index(vector, int(record.vectorid))
            vstore.save_index(FAISS_outfile)

            return Response({
                "message": f"UUID {uuid_str}의 임베딩 벡터가 성공적으로 업데이트되었습니다.",
                "status": "SUCCESS"
            })

        except Vecmanager.DoesNotExist:
            return Response({"message": f"UUID {uuid_str} 에 해당하는 레코드를 찾을 수 없습니다.", "status": "FAIL"},
                            status=status.HTTP_404_NOT_FOUND)
        except ValueError:
            return Response({"message": "UUID 형식이 잘못되었습니다.", "status": "FAIL"},
                            status=status.HTTP_400_BAD_REQUEST)        


    
    @action(detail=False, methods=['delete'], url_path='delete-by-uuid/(?P<uuid_str>[0-9a-f-]+)')
    def delete_by_uuid(self, request, uuid_str=None):
        try:
            # Validate UUID format
            uuid_obj = UUID(uuid_str, version=4)

            # Attempt to retrieve and delete the record
            record = Vecmanager.objects.get(personid=uuid_obj)
            vstore, FAISS_outfile = FAISS_server_start(record.user)
            vectorid = record.vectorid
            record.delete()            

            # Optionally remove from FAISS index if needed
            res = vstore.delete_vec_from_index(vectorid)            
            if res == 1:
                vstore.save_index(FAISS_outfile)
                return Response({
                    'message': f"personid {uuid_str} (vectorid {vectorid}) 삭제되었습니다",
                    'status': "SUCCESS"
                }, status=status.HTTP_200_OK)
            else:
                return Response({'message': f"삭제할 사용자 {uuid_str}의 정보가 존재하지 않습니다", 'status': "FAIL"}, status=status.HTTP_404_NOT_FOUND)

        except Vecmanager.DoesNotExist:
            return Response({
                'message': f"personid {uuid_str} 에 해당하는 레코드가 존재하지 않습니다.",
                'status': "FAIL"
            }, status=status.HTTP_404_NOT_FOUND)

        except ValueError:
            return Response({
                'message': "잘못된 UUID 형식입니다.",
                'status': "FAIL"
            }, status=status.HTTP_400_BAD_REQUEST)



class SearchViewSet(viewsets.ModelViewSet):
    queryset = Searchmanager.objects.all()
    serializer_class = SearchSerializer

    def create(self, request, *args, **kwargs):

        username = request.POST['user']
        vstore, _ = FAISS_server_start(username)
        representation = request.POST['embedvec']        
        represent_list = literal_eval(representation) 

        ######## similarity search one-by-one ###########
        results = []
        for vec in represent_list:
            vector = np.array(vec, dtype=np.float32).reshape(1, -1)
            result = vstore.search_index(vector, 1)
            results.append(result)

        ####### similarity search at once #############

        vectors = np.array(represent_list, dtype=np.float32)
        result2 = vstore.search_index(vectors, 1)

        # print(results)

        return JsonResponse(result2)


    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

 