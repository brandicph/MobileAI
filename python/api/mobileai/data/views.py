from rest_framework import viewsets
from mobileai.data.models import CellInfo, CellInfoLte, CellIdentityLte, CellSignalStrengthLte
from mobileai.data.serializers import CellInfoSerializer, CellInfoLteSerializer, CellIdentityLteSerializer, CellSignalStrengthLteSerializer


class CellInfoViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows cellinfo to be viewed or edited.
    """
    queryset = CellInfo.objects.all().order_by('-created')
    serializer_class = CellInfoSerializer


class CellInfoLteViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows CellInfoLte to be viewed or edited.
    """
    queryset = CellInfoLte.objects.all().order_by('-created')
    serializer_class = CellInfoLteSerializer

class CellIdentityLteViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows CellIdentityLte to be viewed or edited.
    """
    queryset = CellIdentityLte.objects.all().order_by('-created')
    serializer_class = CellIdentityLteSerializer

class CellSignalStrengthLteViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows CellSignalStrengthLte to be viewed or edited.
    """
    queryset = CellSignalStrengthLte.objects.all().order_by('-created')
    serializer_class = CellSignalStrengthLteSerializer



