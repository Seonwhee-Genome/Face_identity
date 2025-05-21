from django.shortcuts import render

# Create your views here.
import requests, os, logging
from django.http import HttpResponse, Http404, FileResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status, viewsets

from .serializers import AISerializer
from .models import AIarchive


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger(__name__)


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
            logger.error(f'model {archive.modelid} does not exist!')
            return Response(
                {'error': f'해당 AI 모델이 서버에 존재하지 않습니다.'},
                status=status.HTTP_404_NOT_FOUND
            )     
        
        try:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            logger.debug(f'file {filename} at {file_path} will be downloaded')
            response = FileResponse(
                open(file_path, 'rb'),
                as_attachment=True,
                filename=filename
            )
            response['Content-Length'] = str(file_size) # ensure Content-Length included in the Response header
            return response
        except Exception as e:
            logger.error(str(e))
            return Response(
                {'error': f'Could not serve file: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
