from django.contrib.auth.models import User, Group
from .models import Entity, Location
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')


class EntitySerializer(serializers.HyperlinkedModelSerializer):
    locations = serializers.HyperlinkedIdentityField(
        view_name='locations-list',
        lookup_url_kwarg='entity_pk'
    )

    class Meta:
        model = Entity
        fields = '__all__'

class LocationSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()
    url = serializers.ReadOnlyField()

    class Meta:
        model = Location
        fields = '__all__'
