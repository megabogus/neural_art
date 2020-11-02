from django.contrib import admin
from artfilter.models import ArtFilter

# Register your models here.

@admin.register(ArtFilter)
class ArtFilterAdmin(admin.ModelAdmin):
    save_on_top = True
    list_display = ('title',)
