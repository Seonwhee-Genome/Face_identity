from django.shortcuts import render

# Create your views here.
from .serializers import VecSerializer, SearchSerializer
from .models import Vecmanager, Searchmanager, VecImage, SearchImage
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
from django.conf import settings
from rest_framework import viewsets
from rest_framework.decorators import action
from django.core.files import File
from django.db.models import Max
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
THRESHOLD = 5.0

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
        pid = request.POST['personid']
        represent_list = literal_eval(representation)
        request.data['embedvec'] = represent_list

        # 🔸 Generate next available vectorid
        max_vectorid = Vecmanager.objects.aggregate(Max('vectorid'))['vectorid__max']
        next_vectorid = 1 if max_vectorid is None else max_vectorid + 1
        request.data['vectorid'] = next_vectorid

        request.data._mutable = False

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        vec_obj = serializer.save(user=username)

        # Save multiple images
        images = request.FILES.getlist('images')
        for img in images:
            VecImage.objects.create(vecmanager=vec_obj, image=img)
            
        # 🔸 Add to vector index
        vector = np.array(represent_list, dtype=np.float32).reshape(1, -1)
        vstore.add_vec_to_index(vector, int(next_vectorid))
        vstore.save_index(FAISS_outfile)

        return Response({'message': f"사용자 {pid}의 정보가 등록되었습니다", 'status': "SUCCESS"}, status=http_codes[201])
        

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
                'imgfilename' : entry.imgfilename,
                'modelid' : entry.modelid,
                'created_at': entry.created_at.isoformat(),
                'current vectorids': vstore.all_ids
            })
            del(vstore)
            
        return Response({'count': len(data), 'results': data})     

            

    @action(detail=False, methods=['post'], url_path='upsert')
    def upsert_vecmanager(self, request):
        try:
            pid = request.data.get("personid", None)
            imgfile = request.data.get("imgfilename", None)
            if not pid:
                return Response({"message": "personid 필드는 필수입니다.", "status": "FAIL"},
                                status=http_codes[400])
            if not imgfile:
                return Response({"message": "imagefilename 필드는 필수입니다.", "status": "FAIL"},
                                status=http_codes[400])

            user = request.data.get("user", "AnonymousUser")            
            embedvec_input = request.data.get("embedvec", None)

            vstore, FAISS_outfile = FAISS_server_start(user)

            if embedvec_input is None:
                return Response({"message": "embedvec 필드는 필수입니다.", "status": "FAIL"},
                                status=http_codes[400])

            if isinstance(embedvec_input, str):
                try:
                    embedvec = literal_eval(embedvec_input)
                except Exception:
                    return Response({"message": "embedvec 문자열을 리스트로 변환할 수 없습니다.", "status": "FAIL"},
                                    status=http_codes[400])
            else:
                embedvec = embedvec_input

            if not isinstance(embedvec, list) or len(embedvec) != 512:
                return Response({"message": "embedvec은 512차원의 리스트여야 합니다.", "status": "FAIL"},
                                status=http_codes[400])

            vector = np.array(embedvec, dtype=np.float32).reshape(1, -1)

            try:
                # Try update                
                record = Vecmanager.objects.get(personid=pid)
                record.embedvec = embedvec
                record.user = user

                oldimgpath = os.path.join(settings.MEDIA_ROOT, 'uploads', record.imgfilename)
                os.remove(oldimgpath)
                print(f"old image file {record.imgfilename} deleted")
                record.imgfilename = imgfile
                
                record.save()
                
                vstore.add_vec_to_index(vector, int(record.vectorid))

                # Save multiple images
                images = request.FILES.getlist('images')
                for img in images:
                    VecImage.objects.create(vecmanager=vec_obj, image=img)
                
            
                action_type = "updated"
            except Vecmanager.DoesNotExist:
                # Create new                
                max_vectorid = Vecmanager.objects.aggregate(Max('vectorid'))['vectorid__max']
                next_vectorid = 1 if max_vectorid is None else max_vectorid + 1                

                record = Vecmanager.objects.create(
                    personid=pid,
                    user=user,
                    vectorid=next_vectorid,
                    embedvec=embedvec
                )

                vstore.add_vec_to_index(vector, int(next_vectorid))
                action_type = "created"

            vstore.save_index(FAISS_outfile)

            return Response({
                "message": f"PersonID {pid} 로 레코드를 성공적으로 {action_type} 했습니다.",
                "personid": record.personid,
                "status": "SUCCESS"
            }, status=http_codes[201])

        except Exception as e:
            return Response({
                "message": f"서버 오류: {str(e)}",
                "status": "FAIL"
            }, status=http_codes[500])
        
            

    @action(detail=False, methods=['delete'], url_path='delete-by-personid/(?P<personid>[^/]+)')
    def delete_by_personid(self, request, personid):
        try:
            # Attempt to retrieve and delete the record
            record = Vecmanager.objects.get(personid=personid)
            vstore, FAISS_outfile = FAISS_server_start(record.user)
            vectorid = record.vectorid
            imgpath = os.path.join(settings.MEDIA_ROOT, 'uploads', record.imgfilename)
            os.remove(imgpath)
            logger.debug(f"image file {record.imgfilename} deleted")
            record.delete()

            # Optionally remove from FAISS index
            res = vstore.delete_vec_from_index(vectorid)
            if res == 1:
                vstore.save_index(FAISS_outfile)
                return Response({
                    'message': f"personid {personid} (vectorid {vectorid}) 삭제되었습니다",
                    'status': "SUCCESS"
                }, status=http_codes[200])
            else:
                return Response({
                    'message': f"삭제할 사용자 {personid}의 벡터 정보가 존재하지 않습니다",
                    'status': "FAIL"
                }, status=http_codes[404])
        except Vecmanager.DoesNotExist as dne:
            logger.error(dne)
            return Response({
                'message': f"personid {personid} 에 해당하는 레코드가 존재하지 않습니다.",
                'status': "FAIL"
            }, status=http_codes[404])

        except ValueError as ve:
            logger.error(ve)
            return Response({
                'message': "잘못된 ID 형식입니다.",
                'status': "FAIL"
            }, status=http_codes[400])

        except FileNotFoundError as fe:
            logger.error(fe)
            return Response({
                    'message': f"삭제할 사용자 {personid}의 이미지 파일이 존재하지 않습니다",
                    'status': "FAIL"
                }, status=http_codes[404])



class SearchViewSet(viewsets.ModelViewSet):
    queryset = Searchmanager.objects.all()
    serializer_class = SearchSerializer
    lookup_field = 'searchid'

    def create(self, request, *args, **kwargs):

        try:
            request.data._mutable = True
            username = request.POST['user'] 
            sid = request.POST['searchid']
            vstore, _ = FAISS_server_start(username)
            representation = request.POST['embedvec']        
            represent_list = literal_eval(representation)            

            # Save multiple images
            images = request.FILES.getlist('images')
            imgPath = request.POST['imgfilename']            
            

            ######## similarity search one-by-one ###########
            img_results = []
            results = []
            for vec in represent_list:
                vector = np.array(vec, dtype=np.float32).reshape(1, -1)
                result, code = vstore.search_index(vector, 2, THRESHOLD)
                results.append(result)

            logger.debug(represent_list[0])
                

            ####### similarity search at once #############

            vectors = np.array(represent_list, dtype=np.float32)
            logger.debug("Try to do similarity search")
            result2, code2 = vstore.search_index(vectors, 2, THRESHOLD)            
            logger.debug("개별 프레임의 유사도")
            logger.debug(results)
            logger.debug("종합 유사도")
            logger.debug(result2)

            entry1 = Vecmanager.objects.get(personid=result2['top 1 id'])
            entry2 = Vecmanager.objects.get(personid=result2['top 2 id'])            

            data = {
                "user": username,
                "searchid" : sid,
                "top1pid": result2['top 1 id'],
                "top2pid": result2['top 2 id'],
                "imgfilename": imgPath,
                "embedvec" : represent_list[0],
                'modelid' : request.POST['modelid'],
                'sim_imgfile1' : entry1.imgfilename,
                'sim_imgfile2' : entry2.imgfilename,                
                'distance1' : result2['top 1 distance'],
                'distance2' : result2['top 2 distance'],
                'identify' : True if result2['status'] == 'IDENTIFIED' else False,
            }
            
            serializer = self.get_serializer(data=data)
            serializer.is_valid(raise_exception=True)
            search_obj = serializer.save()    

            for img in images:
                SearchImage.objects.create(searchmanager=search_obj, image=img)

            # Path to your image file under MEDIA_ROOT
            imgpath1 = os.path.join(settings.MEDIA_ROOT, 'uploads', entry1.imgfilename)
            imgpath2 = os.path.join(settings.MEDIA_ROOT, 'uploads', entry2.imgfilename)
            img_query = os.path.join(settings.MEDIA_ROOT, 'uploads', imgPath)

            sm = Searchmanager.objects.last()
            with open(imgpath1, 'rb') as f:
                # Save file with a relative name to avoid SuspiciousFileOperation exception by Django
                # Django is very strict about file paths.
                sm.sim_image1.save('auto_test1.jpg', File(f)) # ✅ Use just the filename, not full path
                sm.save()
                
            with open(imgpath2, 'rb') as f:           
                sm.sim_image2.save('auto_test2.jpg', File(f))
                sm.save()

            with open(img_query, 'rb') as f:           
                sm.qimage.save('auto_test3.jpg', File(f))
                sm.save()

            return Response(result2, status=http_codes[code2])
            

        except MultiValueDictKeyError as me:
            logger.error(me)
            return Response({"message": f"API 입력값 중에 {me}이 누락되었습니다."}, status=http_codes[400])


    def partial_update(self, request, *args, **kwargs):
        searchid = kwargs.get('searchid')

        # Get all instances that match the searchid
        instances = Searchmanager.objects.filter(searchid=searchid)

        if not instances.exists():
            return Response(
                {"message": f"Search ID '{searchid}'의 안면 검색 기록이 없습니다. 따라서 평가를 내릴 수 없습니다.", "status": "NOT_FOUND"},
                status=http_codes[404]
            )

        # Only update 'correct' field if it's in the request
        if 'correct' in request.data:
            instances.update(correct=request.data['correct'])  # Bulk update
            return Response(
                {
                    "message": f"Search ID {searchid}의 안면인식 결과의 평가를 성공적으로 업데이트 했습니다.",
                    "status": "SUCCESS"
                },
                status=http_codes[200]
            )
        return Response(
            {"message": "'correct' 필드가 요청에 포함되어야 합니다.", "status": "FAIL"},
            status=http_codes[400]
            )


    



 