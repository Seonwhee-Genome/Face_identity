from django.shortcuts import render

# Create your views here.
import requests, os
from django.http import HttpResponse, Http404, FileResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status, viewsets

from .serializers import AISerializer
from .models import AIarchive


class DownloadViewSet(viewsets.ModelViewSet):

    queryset = AIarchive.objects.all()
    serializer_class = AISerializer
    lookup_field = 'modelid'

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.username)

    @action(detail=True, methods=['get'], url_path='download')
    def download(self, request, modelid=None):
        archive = self.get_object()
        file_path = archive.url  # e.g., /data/ai_archive/filename.tflite
        
        if not os.path.exists(file_path):
            raise Http404("File does not exist.")

        filename = os.path.basename(file_path)
        
        try:
            print(file_path)
            return FileResponse(
                open(file_path, 'rb'),
                as_attachment=True,
                filename=filename
            )
        except Exception as e:
            return Response(
                {'error': f'Could not serve file: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
