from channels_api.bindings import ResourceBinding

from .models import UserEntity, Measurement
from .serializers import QuestionSerializer

class MeasurementBinding(ResourceBinding):

    model = Measurement
    stream = "measurements"
    serializer_class = MeasurementSerializer
    queryset = Measurement.objects.all()