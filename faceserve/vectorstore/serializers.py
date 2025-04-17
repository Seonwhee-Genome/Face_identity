from rest_framework.serializers import ModelSerializer, ReadOnlyField
from .models import Vecmanager

class VecSerializer(ModelSerializer):
    
    class Meta:
        model = Vecmanager
        fields = (
            'userid',
            'personid',
            'embedvec',
            'created_at',
        )
        read_only_fields = ('userid', 'personid', 'created_at',)