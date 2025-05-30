from django.db import models

# Create your models here.

class AIarchive(models.Model):
    user = models.CharField(max_length=200)
    modelid = models.CharField(max_length=30)
    modelname = models.CharField(max_length=500)
    url = models.CharField(max_length=5000)
    version = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'model {self.modelname}(id: {self.modelid}, version: {self.version}) created at {self.created_at}'