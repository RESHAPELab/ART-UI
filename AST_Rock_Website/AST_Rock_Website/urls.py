"""
URL configuration for AST_Rock_Website project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('dashboard.urls')),
    path('landing/', TemplateView.as_view(template_name='landing.html'), name='landing'),
    path('generic/', TemplateView.as_view(template_name='generic.html'), name='generic'),
    path('elements/', TemplateView.as_view(template_name='elements.html'), name='elements'),
    path('get_started/', TemplateView.as_view(template_name='get_started.html'), name='get_started'),
    path('accounts/', include('allauth.urls')),
]