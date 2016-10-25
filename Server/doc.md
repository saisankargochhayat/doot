Challenge:
To make a module to identify Indian Sign Language's sign for A-Z (excluding J and Z)

Features List:
1)Pinch Strength
2)Grab Strength
3)Thumb pointing direction x axis
4)Thumb pointing direction y axis
5)Thumb pointing direction z axis
6)Index pointing direction x axis
.
.
17)Pinky pointing direction z axis
18)Angle between metacarpal bone and proximal bone of thumb
19)Angle between proximal bone and intermediate bone of thumb
20)Angle between metacarpal bone and proximal bone of index finger
21)Angle between proximal bone and intermediate bone of index finger
.
.
27)Angle between proximal bone and intermediate bone of pinky
28)Angle between thumb and index
29)Angle between index and middle
30)Angle between middle and ring
31)Angle between ring and pinky


label list:
A,B,C,D,E,F,G,H,I,K,L ..... W,X,Y


Different accuracy lists for different Classifiers:
(Models were trained from data collected from 7 person (Alice , Anisha , Asutosh , Rishav , Sai , Sandy , Sohini))
[4 boys , 3 girls]

Hand length(Tip to tip distance from thumb to pinky when extended)
varies from 17cm to 25.8cm



KNNeighbours Classifier:
100% accuracy for Rishav's Data(Test)
93.1% accuracy for Asutosh's Data(Test1)
100% accuracy for Adwesh's Data (Test2)
98.27% accuracy for Sripad's Data(Test3)




SVM Classifier:
100% accuracy for Rishav's Data(Test)
88.5% accuracy for Asutosh's Data(Test1)
98.8% accuracy for Adwesh's Data (Test2)
88.1% accuracy for Sripad's Data(Test3)

DecisionTree Classifier:

74.83% accuracy for Rishav's Data(Test)
99.93% accuracy for Asutosh's Data(Test1)
73.74% accuracy for Adwesh's Data (Test2)
73.88% accuracy for Sripad's Data(Test3)


Stochastic Gradient Descent:
76.6% accuracy for Rishav's Data(Test)
74.23% accuracy for Asutosh's Data(Test1)
74.89% accuracy for Adwesh's Data (Test2)
74.65% accuracy for Sripad's Data(Test3)
