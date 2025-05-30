from rest_framework.serializers import ModelSerializer, ReadOnlyField
from .models import Vecmanager, Searchmanager, VecImage, SearchImage

class VecImageSerializer(ModelSerializer):
    class Meta:
        model = VecImage
        fields = ['id', 'image', 'uploaded_at']


class VecSerializer(ModelSerializer):
    images = VecImageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Vecmanager
        fields = (
            'id',
            'user',
            'personid',
            'vectorid',
            'embedvec',
            'images',
            'created_at',
        )
        read_only_fields = ('id', 'user', 'created_at',)


class SearchImageSerializer(ModelSerializer):
    class Meta:
        model = SearchImage
        fields = ['id', 'image', 'uploaded_at']


class SearchSerializer(ModelSerializer):
    images = SearchImageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Searchmanager
        fields = (
            'id',
            'user',
            'embedvec',
            'images',
            'created_at',
        )
        read_only_fields = ('user', 'created_at',)

