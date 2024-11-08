from django.urls import path
from . import views
from django.urls import path

urlpatterns = [
    path("", views.index, name="index"),
    path("loading/", views.splash_screen, name="splash_screen"),
    path("view_repository/", views.repositories_by_link, name="view_repository"),
    path(
        "results/<str:username>/<str:repo_name>/",
        views.render_issues_results,
        name="render_issues_results",
    ),
    path("task-status/<str:repo_name>/", views.task_status, name="task_status"),
    path("versions", views.get_CoreEngine_version, name="version_check"),
]
