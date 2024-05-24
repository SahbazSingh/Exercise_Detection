from django.contrib import admin
from gymAIapp.models import Contact, BicepCurlResult, SquatResult, ShoulderLateralRaiseResult, BackPullDownResult, ChestCablePullResult


# Register your models here.
admin.site.register(Contact)
#admin.site.register(BicepCurlResult)

@admin.register(BicepCurlResult)
class BicepCurlResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_time', 'total_curls', 'feedback', 'processed_video_filename')  # Fields to display in the list view
    list_filter = ('analysis_time',)  # Fields to filter by in the sidebar
    search_fields = ('feedback', 'form_errors')  # Fields to search by in the search bar

@admin.register(SquatResult)
class SquatResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_time', 'total_squats', 'feedback', 'processed_video_filename')  # Customize as needed
    list_filter = ('analysis_time',)  # Customize as needed
    search_fields = ('feedback', 'form_errors')  # Customize as needed

@admin.register(ShoulderLateralRaiseResult)
class ShoulderLateralRaiseResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_time', 'total_reps', 'feedback', 'processed_video_filename')  # Adjusted to display total_reps
    list_filter = ('analysis_time',)  # Fields to filter by in the sidebar remain the same
    search_fields = ('feedback', 'form_errors')  # Fields to search by in the search bar remain the same

@admin.register(BackPullDownResult)
class BackPullDownResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_time', 'total_reps', 'feedback', 'processed_video_filename')  # Fields to display in the list view
    list_filter = ('analysis_time',)  # Fields to filter by in the sidebar
    search_fields = ('feedback', 'form_errors')  # Fields to search by in the search bar

@admin.register(ChestCablePullResult)
class ChestCablePullResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_time', 'total_reps', 'feedback', 'processed_video_filename')  # Fields to display in the admin list view
    list_filter = ('analysis_time',)  # Fields to filter by in the sidebar
    search_fields = ('feedback', 'form_errors')  # Fields to search by in the admin search bar