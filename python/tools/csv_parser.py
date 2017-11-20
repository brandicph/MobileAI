import csv

CSV_FILEPATH = '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv'

with open(CSV_FILEPATH) as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',', quotechar='"')#csv.DictReader(csvfile)
    for row in file_reader:
        time = row[0]
        cell_id = row[5]
        rssi = row[7]
        pci = row[8]
        rsrp = row[9]
        rsrq = row[10]
        print('time: {} | cell_id: {} | rssi: {} | pci: {} | rsrp: {} | rsrq: {}'.format(time, cell_id, rssi, pci, rsrp, rsrq))
        #print(', '.join(row))