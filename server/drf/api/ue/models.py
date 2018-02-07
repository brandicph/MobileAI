from django.db import models

# Create your models here.


class UserEntity(models.Model):
    """International mobile subscriber identity (IMSI)
    IMSI:310150123456789
    MCC | 310 | USA
    MNC | 150 | AT&T Mobility
    MSIN | 123456789
    """
    IMSI = models.CharField(max_length=100, help_text='International mobile subscriber identity (IMSI), ex. 310150123456789')
    """International Mobile Equipment Identity (IMEI)
    35-209900-176148-1
    IMEISV code 35-209900-176148-23
    TAC: 35-2099 - issued by the BABT (code 35) with the allocation number 2099
    FAC: 00 - indicating the phone was made during the transition period when FACs were being removed.
    SNR: 176148 - uniquely identifying a unit of this model
    CD: 1 so it is a GSM Phase 2 or higher
    SVN: 23 - The "software version number" identifying the revision of the software installed on the phone. 99 is reserved.
    """
    IMEI = models.CharField(max_length=100, help_text='International Mobile Equipment Identity (IMEI), ex. 35-209900-176148-1')
    """Name
    """
    name = models.CharField(max_length=100, help_text='Name of the device, ex. Samsung Galaxy S7')
    """Timestamps
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('IMSI', 'IMEI')
        ordering = ['name']

    def __str__(self):
        return '%s - %s' % (self.name, self.IMEI)



class Measurement(models.Model):
    """User Entity
    """
    user_entity = models.ForeignKey(UserEntity, on_delete=models.CASCADE, related_name='measurements', blank=False, help_text='User Entity used for measuring.')
    """Key
    """
    key = models.CharField(max_length=100, help_text='Key of the measurement, ex. RSRP[0] for RSRP at antenna 0.')
    """Value
    """
    value = models.CharField(max_length=100, default=None, help_text='Value of the measurement, ex. -100')
    """Unit
    """
    unit = models.CharField(max_length=100, default=None, help_text='Unit of the measurement, ex. dBm')
    """Timestamps
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return '%s: %s' % (self.key, self.value)