from rest_framework.serializers import ModelSerializer, ReadOnlyField
from .models import Vecmanager, Searchmanager, Managemanager

class VecSerializer(ModelSerializer):
    
    class Meta:
        model = Vecmanager
        fields = (
            'user',
            'personid',
            'embedvec',
            'created_at',
        )
        read_only_fields = ('user', 'personid', 'created_at',)


class SearchSerializer(ModelSerializer):
    
    class Meta:
        model = Searchmanager
        fields = (
            'user',
            'embedvec',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)


class ManageSerializer(ModelSerializer):
    
    class Meta:
        model = Managemanager
        fields = (
            'user',
            'personid',
            'embedvec',
            'command',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)