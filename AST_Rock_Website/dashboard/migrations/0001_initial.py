# Generated by Django 5.0.6 on 2024-06-23 23:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="GPTMessage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("system_content", models.TextField()),
                ("user_content", models.TextField()),
                ("assistant_content", models.TextField()),
            ],
        ),
    ]
