from django.db import models


class File(models.Model):
    file = models.FileField(blank=False, null=False)
    modelCategory = models.TextField() 
    def __str__(self):
        return self.file.name
