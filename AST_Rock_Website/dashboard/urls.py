from django.urls import path
from . import views
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('', views.home, name='home'),
    path('your_dashboard/', views.your_dashboard, name='your_dashboard'),
    path('your_repositories/', views.your_repositories, name='your_repositories'),
    path('repositories_by_link/', views.repositories_by_link, name='repositories_by_link'),
    path('', auth_views.LogoutView.as_view(), name='logout'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('repositories/<str:repo_name>/', views.repo_detail, name='repo_detail'),
]

