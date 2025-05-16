from django.shortcuts import render

# Create your views here.
from .serializers import AISerializer
from .models import AIarchive
from rest_framework import viewsets


class DownloadViewSet(viewsets.ModelViewSet):
    queryset = AIarchive.objects.all()
    serializer_class = AISerializer

    def perform_create(self, serializer):
        # Automatically assign the user (you can update this logic as needed)
        serializer.save(user=self.request.user.username)