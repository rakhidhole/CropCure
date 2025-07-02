from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import DetectionHistory

# Inline for DetectionHistory
class DetectionHistoryInline(admin.StackedInline):
    model = DetectionHistory
    extra = 0
    readonly_fields = ('uploaded_image', 'disease_name', 'confidence', 'symptoms', 'treatment', 'detected_at')
    can_delete = False

# Extend the default UserAdmin
class CustomUserAdmin(BaseUserAdmin):
    inlines = [DetectionHistoryInline]

# Unregister the default User admin and register the customized one
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

# Optionally still register DetectionHistory standalone view
@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'disease_name', 'confidence', 'detected_at')
    list_filter = ('disease_name', 'detected_at')
    search_fields = ('user__username', 'disease_name')
