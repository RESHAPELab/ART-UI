from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AST_Rock_Website.settings')

app = Celery('AST_Rock_Website')

# Use REDIS_TLS_URL for secure connection to Redis
app.conf.broker_url = os.getenv('REDIS_TLS_URL')

app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks from all installed apps
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)