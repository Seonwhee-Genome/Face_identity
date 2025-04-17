from django.shortcuts import render

# Create your views here.
from .serializers import VecSerializer
from .models import Vecmanager
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

class RegisterViewSet(viewsets.ModelViewSet):
    queryset = Vecmanager.objects.all()
    serializer_class = VecSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        username = request.POST['userid']
        person = request.POST['personid']
        representation = request.POST['embedvec']
        represent_list = literal_eval(representation)
        vector = np.array(represent_list, dtype=np.float32).reshape(1, -1)
        print(vector)
        return JsonResponse({'response': f"사용자 {person}의 정보가 등록되었습니다"})

    def perform_create(self, serializer):
        serializer.save(userid=self.request.userid)