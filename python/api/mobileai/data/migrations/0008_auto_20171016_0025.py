# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-10-16 00:25
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0007_auto_20171016_0008'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cellidentitylte',
            name='cell_info_lte',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='cell_identity_lte', to='data.CellInfoLte'),
        ),
        migrations.AlterField(
            model_name='cellsignalstrengthlte',
            name='cell_info_lte',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='cell_signal_strength_lte', to='data.CellInfoLte'),
        ),
    ]
