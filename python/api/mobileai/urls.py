from django.conf.urls import url, include
from rest_framework import routers
from mobileai.quickstart import views
from mobileai.data import views as data_views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)
router.register(r'cellinfo', data_views.CellInfoViewSet)
router.register(r'cellinfolte', data_views.CellInfoLteViewSet)
router.register(r'cellidentitylte', data_views.CellIdentityLteViewSet)
router.register(r'cellsignalstrengthlte', data_views.CellSignalStrengthLteViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]