from rest_framework import viewsets
from api.ue.models import UserEntity, Measurement
from api.ue.serializers import UserEntitySerializer, MeasurementSerializer


class UserEntityViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows user entities to be viewed or edited.
    """
    queryset = UserEntity.objects.all().order_by('-name')
    serializer_class = UserEntitySerializer


class MeasurementViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows measurements to be viewed or edited.
    """
    queryset = Measurement.objects.all().order_by('-created_at')
    serializer_class = MeasurementSerializer