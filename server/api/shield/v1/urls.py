"""agilesquad URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from rest_framework import routers
from .alpha import views as views_alpha
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_nested import routers

router = routers.DefaultRouter()
router.register(r'users', views_alpha.UserViewSet)
router.register(r'groups', views_alpha.GroupViewSet)
router.register(r'entities', views_alpha.EntityViewSet)

entity_router = routers.NestedSimpleRouter(router, r'entities', lookup='entity')
entity_router.register(r'locations', views_alpha.LocationViewSet, base_name='locations')


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
# http://www.django-rest-framework.org/api-guide/versioning/
urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^', include(entity_router.urls))
]
