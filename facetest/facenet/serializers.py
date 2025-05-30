from rest_framework.serializers import ModelSerializer, ReadOnlyField
from .models import AIarchive

class AISerializer(ModelSerializer):
    
    class Meta:
        model = AIarchive
        fields = (
            'user',
            'modelid',
            'modelname',
            'url',
            'version',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)