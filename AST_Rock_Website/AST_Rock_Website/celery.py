from __future__ import absolute_import, unicode_literals
import os
import ssl
from celery import Celery
from django.conf import settings
from redis import SSLConnection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AST_Rock_Website.settings')

app = Celery('AST_Rock_Website')

# Use REDIS_TLS_URL for secure connection to Redis
app.conf.broker_url = 'redis://:p155c9e6f4e3b35d7dc1cea4cc35049b9eb52e6d961b10d04485088c365c3fe84@ec2-54-166-204-188.compute-1.amazonaws.com:17489'

app.conf.broker_use_ssl = {
    'ssl_cert_reqs': ssl.CERT_NONE
}

app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks from all installed apps
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)