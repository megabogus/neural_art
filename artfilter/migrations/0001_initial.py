# Generated by Django 2.1.5 on 2019-01-11 22:33

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ArtFilter',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=512, verbose_name='Заголовок')),
                ('filter', models.FileField(upload_to='filter', verbose_name='Фильтр')),
                ('original', models.FileField(upload_to='origin', verbose_name='Оригинал')),
                ('result', models.FileField(upload_to='result', verbose_name='Результат')),
                ('logs', models.TextField(blank=True, null=True, verbose_name='Логи')),
            ],
        ),
    ]