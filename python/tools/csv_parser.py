import csv
from datetime import datetime, timedelta

#CSV_IN_FILEPATH = '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv'
CSV_IN_FILEPATH = '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S.csv'
CSV_OUT_FILEPATH = '../../data/measurement_data.csv'

data = []

""" Notes
    TB: Transport Blocks
    RB: Ressource Blocks
    PRB: Physical Ressource Blocks
"""

with open(CSV_IN_FILEPATH) as csv_in_file:
    file_reader = csv.reader(csv_in_file, delimiter=',', quotechar='"')#csv.DictReader(csvfile)
    next(csv_in_file, None)

    for index, row in enumerate(file_reader):
        time = row[0]
        timestamp = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f') #2017-11-27 14:07:45.440 : %Y-%m-%d %H:%M:%S.%f
        band = row[61].replace('Band ', '').replace(' ', '').split('-')
        bandwidth = row[47]
        cell_id = row[5]
        rssi = row[36]#row[7]
        pci = row[45]#row[8]
        rsrp = row[34]#row[9]
        rsrq = row[35]#row[10]
        ip_thrpt_dl = row[31]
        bytes_transferred_dl = row[74] # TB_avg_size * num_of_TB ~= bytes_transerred_dl
        pdsch_thrpt = row[76]
        cqi0_cqi1 = row[91].replace(' ', '').split('/')
        cycle_count = row[24].replace(' ', '').split('/')
        if len(cycle_count) > 1 and bandwidth != '':
            cycle_id = int(cycle_count[0]) - 1
            
            arr = [
                timestamp,
                #duration.total_seconds() * 1000,#duration,
                cycle_id,
                band,
                int(bandwidth),
                float(rssi),
                float(rsrp),
                float(rsrq),
                int(ip_thrpt_dl),
                int(bytes_transferred_dl),
                int(pdsch_thrpt),
                cqi0_cqi1
            ]

            #data_clustered[data_clustered_idx].append(arr)
            data.append(arr)
            #print(arr)
        #print('time: {} | cell_id: {} | rssi: {} | pci: {} | rsrp: {} | rsrq: {}'.format(time, cell_id, rssi, pci, rsrp, rsrq))
        #print(', '.join(row))
"""
with open(CSV_OUT_FILEPATH) as csv_out_file:
    for row in data:
        print(row)
"""

# sort by cycle and datetime
data.sort(key=lambda r: (r[1], r[0]))

data_clustered_idx = -1
data_clustered = []
data_sorted = []

prev_timestamp = None
for index, value in enumerate(data):
    cycle_id = value[1]

    if cycle_id != data_clustered_idx:
        data_clustered_idx = cycle_id
        data_clustered.append([])
        prev_timestamp = None # since 

    timestamp = value[0]
    prev_timestamp = timestamp if prev_timestamp == None else prev_timestamp
    
    duration = timestamp - prev_timestamp
    prev_timestamp = timestamp
    value.append(duration.total_seconds() * 1000)
    data_sorted.append(value)

    data_clustered[data_clustered_idx].append(value)

    #print(value)

print(data_clustered)