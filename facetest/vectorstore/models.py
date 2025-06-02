from django.db import models

# Create your models here.
from pgvector.django import VectorField

class Vecmanager(models.Model):    
    user = models.CharField(max_length=200)
    personid = models.CharField(max_length=200, unique=True)
    vectorid = models.IntegerField(blank=True, null=True, unique=True)
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)")
    imgfilename = models.CharField(max_length=200, unique=True)
    modelid = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return 'created at {}'.format(self.created_at)
        

class VecImage(models.Model):
    vecmanager = models.ForeignKey(Vecmanager, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Searchmanager(models.Model):
    user = models.CharField(max_length=200, verbose_name="Institution")   
    personid = models.CharField(max_length=200, unique=True, verbose_name="User name")
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)", verbose_name="Embedding Vector of User")
    imgfilename = models.CharField(max_length=200, unique=True, verbose_name = "User image file")
    modelid = models.CharField(max_length=30, verbose_name = "AI model")
    sim_imgfile1 = models.CharField(max_length=200, null=True, verbose_name = "Name of Top 1 similar image")
    sim_imgfile2 = models.CharField(max_length=200, null=True, verbose_name = "Name of Top 2 similar image")
    sim_image1 = models.ImageField(null=True, verbose_name = "Top 1 similar image")
    sim_image2 = models.ImageField(null=True, verbose_name = "Top 2 similar image")
    identify = models.BooleanField(default=False, verbose_name = "Result Positive")
    distance1 = models.FloatField(null=True, verbose_name = "Top 1 distance")
    distance2 = models.FloatField(null=True, verbose_name = "Top 2 distance")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name = "Datetime")

    def __str__(self):
        return 'created at {}'.format(self.created_at)
        

class SearchImage(models.Model):
    searchmanager = models.ForeignKey(Searchmanager, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)