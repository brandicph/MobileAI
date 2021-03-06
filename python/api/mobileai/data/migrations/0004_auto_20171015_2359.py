# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-10-15 23:59
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0003_auto_20171015_2354'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cellidentitylte',
            name='cell_info_lte',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='cell_identity_lte', to='data.CellInfoLte'),
        ),
        migrations.AlterField(
            model_name='cellsignalstrengthlte',
            name='cell_info_lte',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='cell_signal_strength_lte', to='data.CellInfoLte'),
        ),
    ]
