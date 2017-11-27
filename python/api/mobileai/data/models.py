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

"""
class QualiPocNetwork(models.Model):
    name = models.CharField(allow_blank=True) # Network
    operator = models.CharField(allow_blank=True)
    technology = models.CharField(allow_blank=True)
    cell_id = models.IntegerField(blank=True)
    mcc = models.IntegerField(blank=True)
    mnc = models.IntegerField(blank=True)
    lac = models.IntegerField(blank=True)
    dl_earfcn = models.IntegerField(blank=True)
    pci = models.IntegerField(blank=True)
    rssi = models.FloatField(blank=True)
    rsrp = models.FloatField(blank=True)
    rsrq = models.FloatField(blank=True)
    tx_power = models.FloatField(blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class QualiPocServiceTest(models.Model):
    name = models.CharField(allow_blank=True) #Job name
    number = models.IntegerField(blank=True)
    cycle = models.IntegerField(blank=True)
    ip = models.CharField(allow_blank=True)
    kpi_type = models.CharField(allow_blank=True)
    kpi_avg = models.FloatField(blank=True)
    kpi_worst = models.FloatField(blank=True)
    kpi_best = models.FloatField(blank=True)
    kpi_last = models.FloatField(blank=True)
    kpi_intermediate = models.IntegerField(blank=True)
    ip_thrpt_dl = models.IntegerField(blank=True)
    ip_thrpt_ul = models.IntegerField(blank=True)

    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class QualiPocLte(models.Model):
    dl_earfcn = models.IntegerField(blank=True)
    pci = models.IntegerField(blank=True)
    tac = models.IntegerField(blank=True)
    pci = models.IntegerField(blank=True)
    rssi = models.FloatField(blank=True)
    rsrp = models.FloatField(blank=True)
    rsrq = models.FloatField(blank=True)
    sinr_rx_0 = models.FloatField(blank=True)
    sinr_rx_1 = models.FloatField(blank=True)
    rsrp_rx_0 = models.FloatField(blank=True)
    rsrq_rx_0 = models.FloatField(blank=True)
    rssi_rx_0 = models.FloatField(blank=True)
    rsrp_rx_1 = models.FloatField(blank=True)
    rsrq_rx_1 = models.FloatField(blank=True)
    rssi_rx_1 = models.FloatField(blank=True)
    rx_antennas = models.IntegerField(blank=True)
    q_rx_lev_min = models.IntegerField(blank=True)
    p_max = models.IntegerField(blank=True)
    max_tx_power = models.IntegerField(blank=True)
    s_rx_lev = models.IntegerField(blank=True)
    s_intra_search = models.IntegerField(blank=True)
    s_non_intra_search = models.IntegerField(blank=True)
    e_nb = models.IntegerField(blank=True)
    rf_band = models.CharField(allow_blank=True)
    cp_distribution = models.CharField(allow_blank=True)
    timing_advance = models.IntegerField(blank=True)
    emm_state = models.CharField(allow_blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)


class QualiPocResult(models.Model):
    network = models.OneToOneField(QualiPocNetwork, on_delete=models.CASCADE, primary_key=True, blank=True)
    service_test = models.OneToOneField(QualiPocServiceTest, on_delete=models.CASCADE, primary_key=True, blank=True)
    lte = models.OneToOneField(QualiPocLte, on_delete=models.CASCADE, primary_key=True, blank=True)
    time = models.DateTimeField(auto_now_add=True) #yyyy-MM-dd HH:mm:ss.SSS
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('created',)

    def __str__(self):
        return "operator: %s - network: %s" % (self.network.name, self.network.name,)

"""