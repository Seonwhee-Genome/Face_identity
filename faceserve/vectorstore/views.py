from django.shortcuts import render

# Create your views here.
from .serializers import VecSerializer, SearchSerializer
from .models import Vecmanager, Searchmanager
import os, requests, shutil
import logging
import datetime, re
import numpy as np
from ast import literal_eval
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.renderers import JSONRenderer
from django.shortcuts import get_object_or_404
from django.http import Http404,HttpResponse,HttpResponseRedirect, JsonResponse
from django.utils.datastructures import MultiValueDictKeyError

from rest_framework import viewsets
from rest_framework.decorators import action
from django.db.models import Max
from uuid import UUID
from facenet.models import AIarchive
from .faiss_vectorstore import FAISS_FlatL2

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger(__name__)

http_codes = {500: status.HTTP_500_INTERNAL_SERVER_ERROR,
              404: status.HTTP_404_NOT_FOUND,
              400: status.HTTP_400_BAD_REQUEST,
              201: status.HTTP_201_CREATED,
              200: status.HTTP_200_OK}
THRESHOLD = 0.5

def FAISS_server_start(username: str):
    vstore = FAISS_FlatL2(512)
    if os.path.exists(os.path.join(vstore.root, f'faissDB-{username}.index')):
        vstore.load_index(f'faissDB-{username}.index')
        logger.debug(f'지자체 {username}의 FAISS DB load')
    else:
        vstore.create_index()
        logger.debug(f'지자체 {username}의 FAISS DB 생성')
    
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

        try:
            model_ver = request.data['model_ver']
            logger.debug(f"model_version : {model_ver}")
            latest_model = AIarchive.objects.filter(version=model_ver).last()
            print(latest_model)
            if latest_model:
                model_id = latest_model.modelid
                # Optional: also retrieve other info if needed
                model_name = latest_model.modelname
                model_version = latest_model.version
            else:
                model_id = "Unknown"  # or handle case where no models exist
            logger.debug("model_id : %s"%(model_id))
        except MultiValueDictKeyError as ke:
            logger.error(f"no model info passed {ke}")
            model_id = "Unknown"

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=username)
            
        # 🔸 Add to vector index
        vector = np.array(represent_list, dtype=np.float32).reshape(1, -1)
        vstore.add_vec_to_index(vector, int(next_vectorid))
        vstore.save_index(FAISS_outfile)    

        return Response({'message': f"사용자 {uuid}의 정보가 등록되었습니다", 'status': "SUCCESS", 'model_ver': model_id}, status=http_codes[201])
        

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
                'current vectorids': vstore.all_ids,
                'embed model ver' : entry.model_ver 
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
                                status=http_codes[400])

            # Convert string to list if needed
            if isinstance(embedvec_input, str):
                try:
                    embedvec = literal_eval(embedvec_input)
                except Exception as ea:
                    logger.error(ea)
                    return Response({"message": "embedvec 문자열을 리스트로 변환할 수 없습니다.", "status": "FAIL"},
                                    status=http_codes[400])
            else:
                embedvec = embedvec_input

            if not isinstance(embedvec, list) or len(embedvec) != 512:
                return Response({"message": "embedvec은 512차원의 리스트여야 합니다.", "status": "FAIL"},
                                status=http_codes[400])            

            # Update record            
            record.embedvec = embedvec
            model_ver = request.data.get("model_ver", None)
            latest_model = AIarchive.objects.filter(version=model_ver).last()
            if latest_model:
                model_id = latest_model.modelid
                model_name = latest_model.modelname
                model_version = latest_model.version
                record.model_ver = model_ver
            else:
                model_id = "Unknown"  # or handle case where no models exist
            record.save()
            logger.debug("model_id : %s"%(model_id))
            
            logger.debug(f"model_version : {model_ver}")

            # Update FAISS index
            vector = np.array(embedvec, dtype=np.float32).reshape(1, -1)            
            vstore.add_vec_to_index(vector, int(record.vectorid))
            vstore.save_index(FAISS_outfile)            
            

            return Response({
                "message": f"UUID {uuid_str}의 임베딩 벡터가 성공적으로 업데이트되었습니다.",
                "status": "SUCCESS",
                "model_ver" : model_id
            })

        except Vecmanager.DoesNotExist as dne:
            logger.error(dne)
            return Response({"message": f"UUID {uuid_str} 에 해당하는 레코드를 찾을 수 없습니다.", "status": "FAIL"},
                            status=http_codes[404])
        except ValueError as ve:
            logger.error(ve)
            return Response({"message": "UUID 형식이 잘못되었습니다.", "status": "FAIL"},
                            status=http_codes[400])
            

    @action(detail=False, methods=['post'], url_path='upsert')
    def upsert_vecmanager(self, request):
        try:
            uuid_str = request.data.get("personid", None)
            if not uuid_str:
                return Response({"message": "personid 필드는 필수입니다.", "status": "FAIL"},
                                status=http_codes[400])

            try:
                uuid_obj = UUID(uuid_str, version=4)
            except ValueError:
                return Response({"message": "잘못된 UUID 형식입니다.", "status": "FAIL"},
                                status=http_codes[400])


            model_ver = request.data.get("model_ver", None)
            logger.debug(f"model_version : {model_ver}")
            latest_model = AIarchive.objects.filter(version=model_ver).last()
            
            if latest_model:
                model_id = latest_model.modelid
                model_name = latest_model.modelname
                model_version = latest_model.version
            else:
                model_id = "Unknown"  # or handle case where no models exist
            

            user = request.data.get("user", "AnonymousUser")            
            embedvec_input = request.data.get("embedvec", None)
            

            vstore, FAISS_outfile = FAISS_server_start(user)

            if embedvec_input is None:
                return Response({"message": "embedvec 필드는 필수입니다.", "status": "FAIL", "model_ver" : model_id},
                                status=http_codes[400])

            if isinstance(embedvec_input, str):
                try:
                    embedvec = literal_eval(embedvec_input)
                except Exception:
                    return Response({"message": "embedvec 문자열을 리스트로 변환할 수 없습니다.", "status": "FAIL", "model_ver" : model_id},
                                    status=http_codes[400])
            else:
                embedvec = embedvec_input

            if not isinstance(embedvec, list) or len(embedvec) != 512:
                return Response({"message": "embedvec은 512차원의 리스트여야 합니다.", "status": "FAIL", "model_ver" : model_id},
                                status=http_codes[400])

            vector = np.array(embedvec, dtype=np.float32).reshape(1, -1)

            try:
                # Try update                
                record = Vecmanager.objects.get(personid=uuid_obj)
                record.embedvec = embedvec
                record.user = user
                if latest_model:
                    record.model_ver = model_ver
                record.save()
                
                vstore.add_vec_to_index(vector, int(record.vectorid))
                
            
                action_type = "updated"
            except Vecmanager.DoesNotExist:
                # Create new                
                max_vectorid = Vecmanager.objects.aggregate(Max('vectorid'))['vectorid__max']
                next_vectorid = 1 if max_vectorid is None else max_vectorid + 1                

                record = Vecmanager.objects.create(
                    personid=uuid_obj,
                    user=user,
                    vectorid=next_vectorid,
                    embedvec=embedvec
                )

                vstore.add_vec_to_index(vector, int(next_vectorid))
                action_type = "created"

            vstore.save_index(FAISS_outfile)

            return Response({
                "message": f"PersonID {uuid_str} 로 레코드를 성공적으로 {action_type} 했습니다.",
                "personid": record.personid,
                "status": "SUCCESS",
                "model_ver" : model_id
            }, status=http_codes[201])

        except Exception as e:
            return Response({
                "message": f"서버 오류: {str(e)}",
                "status": "FAIL"
            }, status=http_codes[500])


    
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
                }, status=http_codes[200])
            else:
                return Response({'message': f"삭제할 사용자 {uuid_str}의 정보가 존재하지 않습니다", 'status': "FAIL"}, status=status.HTTP_404_NOT_FOUND)

        except Vecmanager.DoesNotExist as dne:
            logger.error(dne)
            return Response({
                'message': f"personid {uuid_str} 에 해당하는 레코드가 존재하지 않습니다.",
                'status': "FAIL"
            }, status=http_codes[404])

        except ValueError as ve:
            logger.error(ve)
            return Response({
                'message': "잘못된 UUID 형식입니다.",
                'status': "FAIL"
            }, status=http_codes[400])



class SearchViewSet(viewsets.ModelViewSet):
    queryset = Searchmanager.objects.all()
    serializer_class = SearchSerializer

    def create(self, request, *args, **kwargs):

        try:
            request.data._mutable = True
            username = request.POST['user']
            vstore, _ = FAISS_server_start(username)
            representation = request.POST['embedvec']        
            represent_list = literal_eval(representation) 

            request.data['embedvec'] = represent_list[0]

            ######## similarity search one-by-one ###########
            results = []
            for vec in represent_list:
                vector = np.array(vec, dtype=np.float32).reshape(1, -1)
                result, code = vstore.search_index(vector, 1, THRESHOLD)
                results.append(result)

            ####### similarity search at once #############
            logger.debug(f"model_version : {request.POST['model_ver']}")

            vectors = np.array(represent_list, dtype=np.float32)
            logger.debug("Try to do similarity search")
            result2, code2 = vstore.search_index(vectors, 1, THRESHOLD)
            logger.debug("개별 프레임의 유사도")
            logger.debug(results)
            logger.debug("종합 유사도")
            logger.debug(result2)

            model_ver = request.data.get("model_ver", None)
            logger.debug(f"model_version : {model_ver}")
            latest_model = AIarchive.objects.filter(version=model_ver).last()
            if latest_model:
                model_id = latest_model.modelid
                model_name = latest_model.modelname
                model_version = latest_model.version
            else:
                model_id = "Unknown"  # or handle case where no models exist    
            logger.debug("model_id : %s"%(model_id))
            
            result2['model_ver'] = model_id
            
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            serializer.save(user=username)


            return Response(result2, status=http_codes[code2])

        except MultiValueDictKeyError as me:
            logger.error(me)
            return Response({"message": f"API 입력값 중에 {me}이 누락되었습니다."}, status=http_codes[400])


    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

 