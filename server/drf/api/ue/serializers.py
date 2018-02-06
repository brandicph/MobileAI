from rest_framework import serializers
from .models import UserEntity, Measurement

class UserEntitySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = UserEntity
        fields = '__all__'


class MeasurementSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Measurement
        fields = '__all__'