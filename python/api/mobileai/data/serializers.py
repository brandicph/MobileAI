from mobileai.data.models import CellInfo
from rest_framework import serializers


class CellInfoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CellInfo
        fields = '__all__'
