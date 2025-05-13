from django.shortcuts import render

# Create your views here.
from .serializers import VecSerializer, SearchSerializer, ManageSerializer
from .models import Vecmanager, Searchmanager, Managemanager
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

from .faiss_vectorstore import FAISS_FlatL2


vstore = FAISS_FlatL2(512)

if os.path.exists(os.path.join(vstore.root, "faissDB.index")):
    vstore.load_index("faissDB.index")
else:
    vstore.create_index()


class RegisterViewSet(viewsets.ModelViewSet):
    queryset = Vecmanager.objects.all()
    serializer_class = VecSerializer

    def create(self, request, *args, **kwargs):
        request.data._mutable = True
        username = request.POST['user']
        person = request.POST['personid']
        representation = request.POST['embedvec']        
        represent_list = literal_eval(representation)   
        request.data['embedvec'] = represent_list
        request.data._mutable = False
        
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        vector = np.array(represent_list, dtype=np.float32).reshape(1,-1)
        vstore.add_vec_to_index(vector, int(person))
        vstore.save_index("faissDB.index")
        
        return JsonResponse({'message': f"사용자 {person}의 정보가 등록되었습니다", 'status': "SUCCESS"})

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class SearchViewSet(viewsets.ModelViewSet):
    queryset = Searchmanager.objects.all()
    serializer_class = SearchSerializer

    def create(self, request, *args, **kwargs):

        username = request.POST['user']
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

        print(results)


        return JsonResponse(result2)


    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ManageViewSet(viewsets.ModelViewSet):
    queryset = Managemanager.objects.all()
    serializer_class = ManageSerializer

    def create(self, request, *args, **kwargs):
        username = request.POST['user']
        person = request.POST['personid']
        command = request.POST['command']
        if command == "replace":
            representation = request.POST['embedvec']
            represent_list = literal_eval(representation)
            return JsonResponse({'message': f"사용자 {person}의 정보가 교체되었습니다", 'status': "SUCCESS"})
        elif command == "remove":
            res = vstore.delete_vec_from_index(person)
            return JsonResponse(res)
            # if res == 1:
            #     return JsonResponse({'message': f"사용자 {person}의 정보가 삭제되었습니다", 'status': "SUCCESS"})
            # else:
            #     return JsonResponse({'message': f"사용자 {person}의 정보가 삭제에 실패하였습니다", 'status': "FAIL"})
        else:
            return JsonResponse({'message': "command를 replace 혹은 remove로 선택하십시오", 'status': "FAIL"})

        
        