import os

root_dirs = ['Experimental Data/FaultyData_1/I1', 'Experimental Data/FaultyData_1/I2', 'Experimental Data/FaultyData_1/I3', 'Experimental Data/Healthy Data']

print('Cleaning and extracting data')

i1 = [open('data/i1_1.csv', 'w'), open('data/i1_2.csv', 'w'), open('data/i1_3.csv', 'w'), open('data/i1_h.csv', 'w')]
i2 = [open('data/i2_1.csv', 'w'), open('data/i2_2.csv', 'w'), open('data/i2_3.csv', 'w'), open('data/i2_h.csv', 'w')]
i3 = [open('data/i3_1.csv', 'w'), open('data/i3_2.csv', 'w'), open('data/i3_3.csv', 'w'), open('data/i3_h.csv', 'w')]

for i in range(4):
    i1[i].write('i1\n')
    i2[i].write('i2\n')
    i3[i].write('i3\n')

for parent_dir in root_dirs:
    for dir in os.listdir(parent_dir):
        if (dir.find('4A') == -1) and (dir.find('5A') == -1) and (dir.find('75') == -1):
            for k in range(30):
                raw = open(parent_dir + '/' + dir +'/test_'+ str(k+1) +'.lvm').readlines()
                for t in range(0, 10000):
                    val = raw[t+23].strip('\n').split('\t')
                    if(parent_dir.find('I1') != -1):
                        if(dir.find('1A') != -1):
                            i1[0].write(val[3] + '\n')
                        elif(dir.find('2A') != -1):
                            i1[1].write(val[3] + '\n')
                        elif(dir.find('3A') != -1):
                            i1[2].write(val[3] + '\n')
                    elif(parent_dir.find('I2') != -1):
                        if(dir.find('1A') != -1):
                            i2[0].write(val[3] + '\n')
                        elif(dir.find('2A') != -1):
                            i2[1].write(val[3] + '\n')
                        elif(dir.find('3A') != -1):
                            i2[2].write(val[3] + '\n')
                    elif(parent_dir.find('I3') != -1):
                        if(dir.find('1A') != -1):
                            i3[0].write(val[3] + '\n')
                        elif(dir.find('2A') != -1):
                            i3[1].write(val[3] + '\n')
                        elif(dir.find('3A') != -1):
                            i3[2].write(val[3] + '\n')
                    elif(parent_dir.find('Healthy') != -1):
                        i1[3].write(val[3] + '\n')
                        i2[3].write(val[3] + '\n')
                        i3[3].write(val[3] + '\n')

for i in range(4):
    i1[i].close()
    i2[i].close()
    i3[i].close()

print('finished')
