test_dir = "TEST DATA(LAB)/TEST DATA(LAB)1/"

i1_3 = open('test_data/i1_3.csv', 'w')
i2_3 = open('test_data/i2_3.csv', 'w')
i3_3 = open('test_data/i3_3.csv', 'w')
i1_h = open('test_data/i1_h.csv', 'w')
i2_h = open('test_data/i2_h.csv', 'w')
i3_h = open('test_data/i3_h.csv', 'w')

for k in range(40):
    raw = open(test_dir + 'test_' + str(k+1) + '.lvm').readlines()
    for t in range(0, 10000):
        val = raw[t+23].strip('\n').split('\t')
        if k<10:
            i1_3.write(val[3] + '\n')
        elif k>=10 and k<20:
            i2_3.write(val[1] + '\n')
        elif k>=20 and k<30:
            i3_3.write(val[3] + '\n')
        else:
            i1_h.write(val[3] + '\n')
            i2_h.write(val[3] + '\n')
            i3_h.write(val[3] + '\n')

i1_3.close()
i2_3.close()
i3_3.close()
i1_h.close()
i2_h.close()
i3_h.close()
