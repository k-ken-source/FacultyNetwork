# Generated by Django 2.0.7 on 2020-10-20 04:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0006_auto_20201018_1024'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='bio',
            field=models.CharField(default='No Bio', max_length=200),
        ),
    ]
