from mobileai.data.models import CellInfo, CellInfoLte, CellIdentityLte, CellSignalStrengthLte
from rest_framework import serializers


class CellInfoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CellInfo
        fields = '__all__'

class CellIdentityLteSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CellIdentityLte
        fields = '__all__'

class CellSignalStrengthLteSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CellSignalStrengthLte
        fields = '__all__'


class CellInfoLteSerializer(serializers.HyperlinkedModelSerializer):
    cell_identity_lte = CellIdentityLteSerializer(read_only=True)
    cell_signal_strength_lte = CellSignalStrengthLteSerializer(read_only=True)
    
    class Meta:
        model = CellInfoLte
        fields = ('id', 'url', 'registered', 'timestamp', 'cell_identity_lte', 'cell_signal_strength_lte', )