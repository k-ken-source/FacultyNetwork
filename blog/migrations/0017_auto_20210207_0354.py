# Generated by Django 2.0.7 on 2021-02-07 03:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0016_auto_20210207_0327'),
    ]

    operations = [
        migrations.AlterField(
            model_name='likes',
            name='Status',
            field=models.CharField(choices=[('like', 'like'), ('unlike', 'unlike')], max_length=6),
        ),
    ]