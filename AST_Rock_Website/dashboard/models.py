from django.db import models
import json


class GPTMessage(models.Model):
    system_content = models.TextField()
    user_content = models.TextField()
    assistant_content = models.TextField()

    def set_system_content(self, data):
        self.system_content = json.dumps(data)

    def get_system_content(self):
        return json.loads(self.system_content)

    def set_user_content(self, data):
        self.user_content = json.dumps(data)

    def get_user_content(self):
        return json.loads(self.user_content)

    def set_assistant_content(self, data):
        self.assistant_content = json.dumps(data)

    def get_assistant_content(self):
        return json.loads(self.assistant_content)
