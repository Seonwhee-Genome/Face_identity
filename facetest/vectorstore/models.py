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
    searchid = models.CharField(max_length=200, verbose_name="Search ID")
    embedvec = VectorField(dimensions=512, help_text="Vector embeddings(Facenet512)", verbose_name="Embedding Vector of User")
    imgfilename = models.CharField(max_length=200, verbose_name = "User image file")
    qimage = models.ImageField(null=True, verbose_name = "Image to identify")
    modelid = models.CharField(max_length=30, verbose_name = "AI model")
    top1pid = models.CharField(null=True, max_length=200, verbose_name="ID of Top 1 similar")
    top2pid = models.CharField(null=True, max_length=200, verbose_name="ID of Top 2 similar")
    sim_imgfile1 = models.CharField(max_length=200, null=True, verbose_name = "Name of Top 1 similar image")
    sim_imgfile2 = models.CharField(max_length=200, null=True, verbose_name = "Name of Top 2 similar image")
    sim_image1 = models.ImageField(null=True, verbose_name = "Top 1 similar image")
    sim_image2 = models.ImageField(null=True, verbose_name = "Top 2 similar image")
    identify = models.BooleanField(default=False, verbose_name = "Positive")
    correct = models.BooleanField(null=True, verbose_name = "Correctly Identified")
    distance1 = models.FloatField(null=True, verbose_name = "Top 1 distance")
    distance2 = models.FloatField(null=True, verbose_name = "Top 2 distance")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name = "Datetime")

    def __str__(self):
        return 'created at {}'.format(self.created_at)

    def image_thumb(self):
        if self.qimage:
            return format_html('<img src="{}" width="100" height="100" />', self.qimage.url)
        return "(No image)"


    def sim_image1_thumb(self):
        if self.sim_image1:
            return format_html('<img src="{}" width="100" height="100" />', self.sim_image1.url)
        return "(No image)"


    def sim_image2_thumb(self):
        if self.sim_image2:
            return format_html('<img src="{}" width="100" height="100" />', self.sim_image2.url)
        return "(No image)"
        
        

class SearchImage(models.Model):
    searchmanager = models.ForeignKey(Searchmanager, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def image_tag(self):
        if self.image:
            return format_html('<img src="{}" width="100" height="100" style="object-fit: cover;" />', self.image.url)
        return "No image"

    image_tag.short_description = 'Preview'