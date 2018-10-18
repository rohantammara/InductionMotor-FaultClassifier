import os

if os.path.exists('i3.csv') or os.path.exists('diff_i1_2.csv') or os.path.exists('y.csv') == False:
    dirs = ['Experimental Data/FaultyData_1/I3/'+str(dir) for dir in os.listdir('Experimental Data/FaultyData_1/I3') if dir.find('3A RL_') != -1]
    for dir in os.listdir('Experimental Data/Healthy Data'):
        if dir.find('75') == -1:
            dirs.append('Experimental Data/Healthy Data/'+dir)

    diff = open('diff_i1_2.csv', 'w')
    i = open('i3.csv', 'w')
    y = open('y.csv', 'w')

    diff.write('diff_i1_2\n')
    i.write('i3\n')
    y.write('y\n')

    for dir in dirs:
        for k in range(30):
            raw = open(dir+'/test_'+ str(k+1) +'.lvm').readlines()

            for t in range(0, 10000):
                val = raw[t+23].strip('\n').split('\t')
                diff.write(val[1] + '\n')
                i.write(val[3] + '\n')
                if(dir.find('load') != -1):
                    y.write(str(0) + '\n')
                else:
                    y.write(str(1) + '\n')
    diff.close()
    i.close()
    y.close()
