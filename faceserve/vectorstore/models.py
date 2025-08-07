from django.db import models

# Create your models here.
import uuid
from pgvector.django import VectorField

class Vecmanager(models.Model):    
    user = models.CharField(max_length=200)
    personid = models.UUIDField(default=uuid.uuid4, unique=True)
    vectorid = models.IntegerField(blank=True, null=True, unique=True)
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)")
    model_ver = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    
    def __str__(self):
        return 'created at {}'.format(self.created_at)


class Searchmanager(models.Model):
    user = models.CharField(max_length=200)    
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)")
    model_ver = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return 'created at {}'.format(self.created_at)