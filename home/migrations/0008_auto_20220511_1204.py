# Generated by Django 3.2.13 on 2022-05-11 06:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0007_auto_20220511_1120'),
    ]

    operations = [
        migrations.CreateModel(
            name='super',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('supervisor', models.CharField(max_length=50, null=True)),
                ('date', models.DateField(null=True)),
                ('time', models.TimeField(null=True)),
            ],
        ),
        migrations.AddField(
            model_name='cat',
            name='token',
            field=models.IntegerField(null=True),
        ),
    ]
