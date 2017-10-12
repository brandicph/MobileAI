from django.db import models


class CellInfo(models.Model):
    CQI = models.IntegerField(blank=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)
