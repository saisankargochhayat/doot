import pandas
import matplotlib.pyplot as plt
dataFrame = pandas.read_csv('../CSV_Data/dataset_3.csv')
import statistics as s
ydata = []
y1data = []
labeldata = []
alphabets = [chr(x) for x in range(97,97+25)]
alphabets.remove('j')
for label in alphabets:
    curr_data = dataFrame['pinchStrength'][dataFrame['label']==label].values
    print(label)
    print(curr_data)
    ydata.append(s.pvariance(curr_data))
    y1data.append(curr_data.mean())
    labeldata.append(ord(label))

plt.xticks(labeldata,alphabets)
plt.plot(labeldata,ydata,color='red')
plt.plot(labeldata,y1data,color='blue')
plt.show()
