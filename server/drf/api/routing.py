from django.conf.urls import url

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from api.ue.consumers import MeasurementConsumer

application = ProtocolTypeRouter({
    # WebSocket chat handler
    "websocket": URLRouter([
            url("^measurement/$", MeasurementConsumer),
    ]),
})