"""shield URL Configuration

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
from django.contrib import admin
from django.urls import path
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static
from .v1 import urls as v1_urls 

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
# http://www.django-rest-framework.org/api-guide/versioning/
urlpatterns = [
    url(r'^', include(v1_urls)),
    #url(r'^v1/', include('shield.v1.urls', namespace='v1')),
    url(r'^admin/', admin.site.urls),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]


if settings.DEBUG is True:
    #urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
