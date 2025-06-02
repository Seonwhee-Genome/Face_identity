from django.contrib import admin

# Register your models here.
from .models import Searchmanager, SearchImage

class SearchImageInline(admin.TabularInline):
    model = SearchImage
    extra = 1

@admin.register(Searchmanager)
class SearchmanagerAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'personid', 'imgfilename', 'modelid', 
        'identify', 'distance1', 'distance2', 'created_at'
    )
    list_filter = ('user', 'identify', 'created_at')
    search_fields = ('personid', 'imgfilename', 'modelid')
    inlines = [SearchImageInline]
    

@admin.register(SearchImage)
class SearchImageAdmin(admin.ModelAdmin):
    list_display = ('searchmanager', 'image', 'uploaded_at')
    list_filter = ('uploaded_at',)
