from rest_framework import viewsets
from mobileai.data.models import CellInfo
from mobileai.data.serializers import CellInfoSerializer


class CellInfoViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows cellinfo to be viewed or edited.
    """
    queryset = CellInfo.objects.all().order_by('-created')
    serializer_class = CellInfoSerializer
