from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploaded_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File {self.file.name} uploaded at {self.uploaded_at}"

class CommentAnalysis(models.Model):
    comment = models.TextField()
    sentiment = models.CharField(max_length=10)
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='analyses')

    def __str__(self):
        return f"Comment: {self.comment[:30]}... Sentiment: {self.sentiment}"
