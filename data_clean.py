import os

if (os.path.exists('data/i3.csv') or os.path.exists('data/i1.csv') or os.path.exists('data/i2.csv') or os.path.exists('data/y.csv')) == False:
    root_dirs = ['Experimental Data/FaultyData_1/I1', 'Experimental Data/FaultyData_1/I2', 'Experimental Data/FaultyData_1/I3', 'Experimental Data/Healthy Data']

    if not os.path.exists('data'):
        os.makedirs('data')

    i1 = open('data/i1.csv', 'w')
    i2 = open('data/i2.csv', 'w')
    i3 = open('data/i3.csv', 'w')
    y = open('data/y.csv', 'w')

    i1.write('i1\n')
    i2.write('i2\n')
    i3.write('i3\n')
    y.write('y\n')

    for parent_dir in root_dirs:
        for dir in os.listdir(parent_dir):
            if (dir.find('4A') == -1) and (dir.find('5A') == -1) and (dir.find('75') == -1):
                for k in range(30):
                    raw = open(parent_dir + '/' + dir +'/test_'+ str(k+1) +'.lvm').readlines()
                    for t in range(0, 10000):
                        val = raw[t+23].strip('\n').split('\t')
                        if(parent_dir.find('I1') != -1):
                            i1.write(val[3] + '\n')
                        elif(parent_dir.find('I2') != -1):
                            i2.write(val[3] + '\n')
                        elif(parent_dir.find('I3') != -1):
                            i3.write(val[3] + '\n')
                            if(dir.find('1A') != -1):
                                y.write(str(1) + '\n')
                            elif(dir.find('2A') != -1):
                                y.write(str(2) + '\n')
                            elif(dir.find('3A') != -1):
                                y.write(str(3) + '\n')
                        elif(parent_dir.find('Healthy') != -1):
                            i1.write(val[3] + '\n')
                            i2.write(val[3] + '\n')
                            i3.write(val[3] + '\n')
                            y.write(str(0) + '\n')


    i1.close()
    i2.close()
    i3.close()
    y.close()
    print('Finished cleaning and extracting.')
