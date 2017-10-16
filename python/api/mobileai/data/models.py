from django.db import models


class CellInfo(models.Model):
    CQI = models.IntegerField(blank=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class CellInfoLte(models.Model):
    registered = models.BooleanField(default=False)
    timestamp = models.DateTimeField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class CellIdentityLte(models.Model):
    cell_info_lte = models.OneToOneField(CellInfoLte, related_name='cell_identity_lte', on_delete=models.CASCADE)
    mcc = models.IntegerField()
    mnc = models.IntegerField()
    ci = models.IntegerField()
    pci = models.IntegerField()
    tac = models.IntegerField()
    earfcn = models.IntegerField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class CellSignalStrengthLte(models.Model):
    cell_info_lte = models.OneToOneField(CellInfoLte, related_name='cell_signal_strength_lte', on_delete=models.CASCADE)
    ss = models.IntegerField()
    rsrp = models.IntegerField()
    rsrq = models.IntegerField()
    rssnr = models.IntegerField()
    cqi = models.IntegerField()
    ta = models.IntegerField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)