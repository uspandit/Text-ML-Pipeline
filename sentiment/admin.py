from django.contrib import admin
from .models import UploadedFile, CommentAnalysis

admin.site.register(UploadedFile)
admin.site.register(CommentAnalysis)
