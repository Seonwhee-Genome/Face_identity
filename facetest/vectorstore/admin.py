# from django.contrib import admin

# # Register your models here.
# from .models import Searchmanager, SearchImage

# class SearchImageInline(admin.TabularInline):
#     model = SearchImage
#     extra = 1

# @admin.register(Searchmanager)
# class SearchmanagerAdmin(admin.ModelAdmin):
#     list_display = (
#         'user', 'personid', 'imgfilename', 'modelid', 
#         'identify', 'distance1', 'distance2', 'created_at'
#     )
#     list_filter = ('user', 'identify', 'created_at')
#     search_fields = ('personid', 'imgfilename', 'modelid')
#     inlines = [SearchImageInline]
    

# @admin.register(SearchImage)
# class SearchImageAdmin(admin.ModelAdmin):
#     list_display = ('searchmanager', 'image', 'uploaded_at')
#     list_filter = ('uploaded_at',)



from django.contrib import admin
from django.utils.html import format_html
from .models import Searchmanager, SearchImage

class SearchImageInline(admin.TabularInline):
    model = SearchImage
    extra = 1
    readonly_fields = ['image_preview']  # Show thumbnail in inline
    fields = ['image', 'image_preview']
    
    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" style="object-fit: cover;" />', obj.image.url) # obj.image.url
        return "No preview"
    image_preview.short_description = "Thumbnail"

@admin.register(Searchmanager)
class SearchmanagerAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'personid', 'imgfilename', 'image_thumb',  
        'identify', 'correct', 'sim_image1_thumb', 'distance1', 'sim_image2_thumb', 'distance2',  'modelid', 'created_at'
    )
    list_filter = ('user', 'identify', 'created_at')
    search_fields = ('personid', 'imgfilename', 'modelid')
    inlines = [SearchImageInline]
    readonly_fields = ['image_thumb', 'sim_image1_thumb', 'sim_image2_thumb']

    def image_thumb(self, obj):
        if obj.qimage:
            return format_html('<img src="{}" width="100" height="100" />', obj.qimage.url)
        return "(No image)"
    image_thumb.short_description = "Image to identify"
    
    def sim_image1_thumb(self, obj):
        if obj.sim_image1:
            return format_html('<img src="{}" width="100" height="100" style="object-fit: cover;" />', obj.sim_image1.url) # obj.sim_image1.url        
        return "no image"
    sim_image1_thumb.short_description = "Top 1 Thumbnail"
    
    def sim_image2_thumb(self, obj):
        if obj.sim_image2:
            return format_html('<img src="{}" width="100" height="100" style="object-fit: cover;" />', obj.sim_image2.url) # 
        
        return "no image"
    sim_image2_thumb.short_description = "Top 2 Thumbnail"

@admin.register(SearchImage)
class SearchImageAdmin(admin.ModelAdmin):
    list_display = ('searchmanager', 'image', 'uploaded_at', 'image_preview')
    readonly_fields = ['image_preview']

    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" style="object-fit: cover;" />', obj.image.url)
        return "-"
    image_preview.short_description = "Thumbnail"
