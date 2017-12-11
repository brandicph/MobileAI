import csv
from datetime import datetime, timedelta
import pandas as pd

#CSV_IN_FILEPATH = '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv'
#CSV_IN_FILEPATH = '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S.csv'
CSV_IN_FILEPATH = '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv'
CSV_OUT_FILEPATH = '../../data/measurement_data.csv'

DATA = []
DATA_CLUSTERED = []
DATA_SORTED = []
#DATA_FRAME = pd.DataFrame(columns=['timestamp', 'cycle', 'band', 'bandwidth', 'rssi', 'rsrp', 'rsrq', 'ip_thrpt_dl', 'bytes_transferred_dl', 'pdsch_thrpt', 'cqi0_cqi1', 'duration'])

""" Notes
    TB: Transport Blocks
    RB: Ressource Blocks
    PRB: Physical Ressource Blocks
"""
def parse(filepath=CSV_IN_FILEPATH, sorted=True, clustered=False):
    with open(filepath) as csv_in_file:
        file_reader = csv.reader(csv_in_file, delimiter=',', quotechar='"')#csv.DictReader(csvfile)
        next(csv_in_file, None)

        for index, row in enumerate(file_reader):
            time = row[0]
            timestamp = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f') #2017-11-27 14:07:45.440 : %Y-%m-%d %H:%M:%S.%f
            band = row[61].replace('Band ', '').replace(' ', '').split('-')
            bandwidth = row[47]
            technology = row[4]
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
            if len(cycle_count) > 1 and technology == 'LTE' and bandwidth != '' and bytes_transferred_dl != '' and ip_thrpt_dl != '':
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

                #DATA_CLUSTERED[DATA_CLUSTERED_idx].append(arr)
                DATA.append(arr)
                #print(arr)
            #print('time: {} | cell_id: {} | rssi: {} | pci: {} | rsrp: {} | rsrq: {}'.format(time, cell_id, rssi, pci, rsrp, rsrq))
            #print(', '.join(row))
    """
    with open(CSV_OUT_FILEPATH) as csv_out_file:
        for row in DATA:
            print(row)
    """

    # sort by cycle and datetime
    DATA.sort(key=lambda r: (r[1], r[0]))

    data_clustered_idx = -1

    prev_timestamp = None
    for index, value in enumerate(DATA):
        cycle_id = value[1]

        if cycle_id != data_clustered_idx:
            data_clustered_idx = cycle_id
            DATA_CLUSTERED.append([])
            prev_timestamp = None # since 

        timestamp = value[0]
        prev_timestamp = timestamp if prev_timestamp == None else prev_timestamp
        
        duration = timestamp - prev_timestamp
        prev_timestamp = timestamp
        value.append(duration.total_seconds() * 1000)
        DATA_SORTED.append(value)

        DATA_CLUSTERED[data_clustered_idx].append(value)

        #print(DATA_FRAME)
        """
        DATA_FRAME.append({
            'timestamp': value[0],
            'cycle': value[1],
            'band': value[2],
            'bandwidth': value[3],
            'rssi': value[4],
            'rsrp': value[5],
            'rsrq': value[6],
            'ip_thrpt_dl': value[7],
            'bytes_transferred_dl': value[8],
            'pdsch_thrpt': value[9],
            'cqi0_cqi1': value[10],
            'duration': value[11]
        }, ignore_index=True)
        """
        #DATA_FRAME.append(value)

        #print(value)

    #print(DATA_CLUSTERED)
    #return DATA_FRAME

    if clustered:
        return DATA_CLUSTERED

    return DATA_SORTED if sorted else DATA


