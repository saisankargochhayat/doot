import csv
def load_file(file_name,data,labels):
    with open(file_name, 'r') as ppd:
        for line in ppd:
            attr = line.split(',')
            data.append(list(map(float, attr[0:31])))
            labels.append(attr[31][0:-1])
def data_loader():
    data = []
    labels = []
    load_file("../old/master_data/Alice/a_f.csv",data,labels)
    load_file("../old/master_data/Alice/g_m.csv",data,labels)
    load_file("../old/master_data/Alice/n_s.csv",data,labels)
    load_file("../old/master_data/Alice/t_y.csv",data,labels)

    load_file("../old/master_data/Anisha/a_f.csv",data,labels)
    load_file("../old/master_data/Anisha/g_m.csv",data,labels)
    load_file("../old/master_data/Anisha/n_s.csv",data,labels)
    load_file("../old/master_data/Anisha/t_y.csv",data,labels)

    load_file("../old/master_data/Asutosh/a_f.csv",data,labels)
    load_file("../old/master_data/Asutosh/g_m.csv",data,labels)
    load_file("../old/master_data/Asutosh/n_s.csv",data,labels)
    load_file("../old/master_data/Asutosh/t_y.csv",data,labels)

    load_file("../old/master_data/Rishav/a_f.csv",data,labels)
    load_file("../old/master_data/Rishav/g_m.csv",data,labels)
    load_file("../old/master_data/Rishav/n_s.csv",data,labels)
    load_file("../old/master_data/Rishav/t_y.csv",data,labels)

    load_file("../old/master_data/Sai/a_f.csv",data,labels)
    load_file("../old/master_data/Sai/g_m.csv",data,labels)
    load_file("../old/master_data/Sai/n_s.csv",data,labels)
    load_file("../old/master_data/Sai/t_y.csv",data,labels)

    load_file("../old/master_data/Sandy/a_f.csv",data,labels)
    load_file("../old/master_data/Sandy/g_m.csv",data,labels)
    load_file("../old/master_data/Sandy/n_s.csv",data,labels)
    load_file("../old/master_data/Sandy/t_y.csv",data,labels)

    load_file("../old/master_data/Sohini/a_f.csv",data,labels)
    load_file("../old/master_data/Sohini/g_m.csv",data,labels)
    load_file("../old/master_data/Sohini/n_s.csv",data,labels)
    load_file("../old/master_data/Sohini/t_y.csv",data,labels)

    load_file("../old/master_data/zairza.csv",data,labels)
    return data,labels
