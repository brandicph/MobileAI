from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from rest_framework import routers
from rest_framework.documentation import include_docs_urls
from api.ue import views

router = routers.DefaultRouter()
router.register(r'userentity', views.UserEntityViewSet)
router.register(r'measurement', views.MeasurementViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^docs/', include_docs_urls(title='Mobile AI API')),
    url(r'^channels-api/', include('channels_api.urls'))
]