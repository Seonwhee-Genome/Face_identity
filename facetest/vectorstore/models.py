from django.db import models

# Create your models here.
from pgvector.django import VectorField

class Vecmanager(models.Model):    
    user = models.CharField(max_length=200)
    personid = models.CharField(max_length=200, unique=True)
    vectorid = models.IntegerField(blank=True, null=True, unique=True)
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)")                         
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return 'created at {}'.format(self.created_at)
        

class VecImage(models.Model):
    vecmanager = models.ForeignKey(Vecmanager, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Searchmanager(models.Model):
    user = models.CharField(max_length=200)   
    personid = models.CharField(max_length=200, unique=True)
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)")                           
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return 'created at {}'.format(self.created_at)
        

class SearchImage(models.Model):
    searchmanager = models.ForeignKey(Searchmanager, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)