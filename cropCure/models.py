from django.db import models

#Create your models here.
from django.db import models
from django.contrib.auth.models import User

class DetectionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_image = models.ImageField(upload_to='detected_uploads/')
    disease_name = models.CharField(max_length=100)
    confidence = models.CharField(max_length=20)
    symptoms = models.TextField()
    treatment = models.TextField()
    detected_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.disease_name} - {self.detected_at.strftime('%Y-%m-%d %H:%M')}"
