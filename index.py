import pandas as pd
import random

data = pd.read_csv('Salary.csv')
index = []
train_index=[]
test_index = []

# data = random.shuffle(data)
YearsExperience = list(data['YearsExperience'])
salary = list(data['Salary'])
m = len(salary)
temp = (m*80)//100

for i in range(0,m):
    index.append(i)

# print(index)
index = random.sample(index,len(index))

train_index = index[0:temp]
test_index = index[-(m-temp):]

# print(test_index)

# print(temp)
x_train=[]
y_train=[]
x_test=[]
y_test=[]

for i in train_index:
    x_train.append(YearsExperience[i])
    y_train.append(salary[i])

# print(len(x_train))
for i in test_index:
     x_test.append(YearsExperience[i])
     y_test.append(salary[i])

def salaryPredictionModel(salary,yearOfExperienced):
    A = 0
    B = 0
    C = 0
    D = 0
    m = len(salary)
    for x in yearOfExperienced:
        A += x
        C += (x*x)
        
    for i in range(0,len(salary)):
        x=yearOfExperienced[i]
        y=salary[i]
        B += y
        D += x*y
    
    a0 = (A*D - B*C)/(A*A - m*C)
    a1 = (A*B - D*m)/(A*A - m*C)
    return a0,a1


def prediction(y_train,x_train,x_test):
    y_predict=[]
    a0,a1 = salaryPredictionModel(y_train,x_train)
    for x in x_test:
          val = a0+a1*x
          y_predict.append(val)
        #   print(val)
    return y_predict

y_prediction = prediction(y_train,x_train,x_test)
# print(y_prediction)
error = 0
total = 0
for i in range(len(y_prediction)):
    error +=float(abs(float(y_prediction[i])-float(y_test[i]))/y_test[i])
    
avg = error/len(y_test)
accuracy = 100-avg*100
print('accuracy is',end=" ")
print(accuracy)


import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_prediction,color='blue')
plt.title('salary vs experience')
plt.xlabel('YearofExperience')
plt.ylabel('Salary')
plt.show()
