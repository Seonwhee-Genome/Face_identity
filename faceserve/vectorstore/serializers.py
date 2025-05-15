from rest_framework.serializers import ModelSerializer, ReadOnlyField
from .models import Vecmanager, Searchmanager

class VecSerializer(ModelSerializer):
    
    class Meta:
        model = Vecmanager
        fields = (
            'user',
            'personid',
            'vectorid',
            'embedvec',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)


class SearchSerializer(ModelSerializer):
    
    class Meta:
        model = Searchmanager
        fields = (
            'user',
            'embedvec',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)

