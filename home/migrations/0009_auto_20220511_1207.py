# Generated by Django 3.2.13 on 2022-05-11 06:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0008_auto_20220511_1204'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cat',
            name='token',
        ),
        migrations.AddField(
            model_name='super',
            name='token',
            field=models.IntegerField(null=True),
        ),
    ]
