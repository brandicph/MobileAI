from django.db import models

# Create your models here.
class Entity(models.Model):
    name = models.CharField(max_length=255)
    imei = models.CharField(max_length=255)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return "{} - {}".format(self.name, self.imei)

class Location(models.Model):
    entity = models.ForeignKey(Entity, on_delete=models.CASCADE)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return "{},{}".format(self.longitude, self.latitude)


class Measurement(models.Model):
    entity = models.ForeignKey(Entity, on_delete=models.CASCADE)
    rsrp = models.FloatField()
    rsrq = models.FloatField()
    rssi = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return "{},{},{}".format(self.rsrp, self.rsrq, self.rssi)
