# Generated by Django 2.1.5 on 2019-01-11 22:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('artfilter', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='artfilter',
            name='result',
            field=models.FileField(blank=True, null=True, upload_to='result', verbose_name='Результат'),
        ),
    ]
