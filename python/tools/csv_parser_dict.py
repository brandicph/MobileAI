import csv

CSV_FILEPATH = '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv'

with open(CSV_FILEPATH) as csvfile:
    file_reader = csv.DictReader(csvfile)
    for row in file_reader:
        time = row['Time']
        cell_id = row['Cell Id']
        rssi = row['RSSI']
        pci = row['PCI']
        rsrp = row['RSRP']
        rsrq = row['RSRQ']
        print(row)
        break
        #print('time: {} | cell_id: {} | rssi: {} | pci: {} | rsrp: {} | rsrq: {}'.format(time, cell_id, rssi, pci, rsrp, rsrq))
        #print(', '.join(row))