from rest_framework import filters, mixins
import django_filters.rest_framework
from django.contrib.auth.models import User, Group
from .models import Entity, Location, Measurement
from rest_framework import viewsets, pagination
from rest_framework.response import Response
from .serializers import UserSerializer, GroupSerializer, EntitySerializer, LocationSerializer, MeasurementSerializer


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    filter_backends = (filters.SearchFilter, django_filters.rest_framework.DjangoFilterBackend,)
    search_fields = ('username', 'email',)
    filter_fields = ('username', 'email',)


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    filter_backends = (filters.SearchFilter, django_filters.rest_framework.DjangoFilterBackend,)
    search_fields = ('name',)
    filter_fields = ('name',)


class EntityViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Entity to be viewed or edited.
    """
    queryset = Entity.objects.all().order_by('-created_at')
    serializer_class = EntitySerializer
    filter_backends = (filters.SearchFilter, django_filters.rest_framework.DjangoFilterBackend,)
    search_fields = '__all__'
    filter_fields = '__all__'


class LocationViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Location to be viewed or edited.
    """
    queryset = Location.objects.all()
    serializer_class = LocationSerializer
    filter_backends = (filters.SearchFilter, django_filters.rest_framework.DjangoFilterBackend,)
    search_fields = '__all__'
    filter_fields = '__all__'

    def get_queryset(self):
        return Location.objects.filter(entity=self.kwargs['entity_pk']).order_by('-created_at')


class MeasurementViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Measurements to be viewed or edited.
    """
    queryset = Measurement.objects.all()
    serializer_class = MeasurementSerializer
    filter_backends = (filters.SearchFilter, django_filters.rest_framework.DjangoFilterBackend,)
    search_fields = '__all__'
    filter_fields = '__all__'

    def get_queryset(self):
        return Measurement.objects.filter(entity=self.kwargs['entity_pk']).order_by('-created_at')
