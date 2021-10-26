import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

task_sigma = 1
N = 80
C = 1.5

def calcLostFunc(y1 ,y2):
    res = []
    if len(y1) == len(y2):
        for i in range(len(y1)):
            res.append((y1[i]-y2[i])*(y1[i]-y2[i]))

    return sum(res)

def writeSetsTofile(x,y,name):
    if len(x) == len(y):
        new_file = open(name, "w")
        for i in range(len(y)):
            st = "" + str(x[i]) + " , " + str(y[i])+"\n"
            new_file.write(st)
        new_file.close()


def f_of_x_withOut_nocice(alpha,x):
    C = 1.5
    return  C*np.exp(alpha*x)

def getTheNewEpslone(sigma):
    return np.random.normal(0,sigma)

def y_with_nocie(alpha,x,sigma):
    return f_of_x_withOut_nocice(alpha,x) + getTheNewEpslone(sigma)


def makeOneYs(x_set,y_func_with_nocice,alpha,sigma):
    y = []
    for x in x_set:
        y.append(y_func_with_nocice(alpha,x,sigma))

    return y


def makeOneYs_no_noice(x_set,y_func_no_nocice,alpha):
    y = []
    for x in x_set:
        y.append(y_func_no_nocice(alpha,x))

    return y

Xs_training = sorted(np.random.uniform(-4, 4, N).reshape(-1, 1))
Xs_test = sorted(np.random.uniform(-4,4,N).reshape(-1,1))

alphas = np.linspace(0.1,2.1,11)


all_Alphas_Ys_Training_to_Xs_trainingSet = []
all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice = []
all_Alphas_Ys_test_to_Xs_testSet = []
all_predicted_Ys_from_Xs_trainingSet = []
all_predicted_Ys_from_Xs_testSet = []




for a in alphas:
    all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice.append(makeOneYs_no_noice(Xs_training, y_func_no_nocice=f_of_x_withOut_nocice, alpha=a))


for a in alphas:
    all_Alphas_Ys_Training_to_Xs_trainingSet.append(makeOneYs(Xs_training, y_func_with_nocice=y_with_nocie, alpha=a, sigma=task_sigma))
    all_Alphas_Ys_test_to_Xs_testSet.append(makeOneYs(Xs_test, y_func_with_nocice=y_with_nocie, alpha=a, sigma=task_sigma))


lr = LinearRegression()
tableOfResults_q2_A_and_B = []
for i in range(len(alphas)):
    writeSetsTofile(Xs_training,all_Alphas_Ys_Training_to_Xs_trainingSet[i],"dataSetfor" + str(i) + ".txt")
    lr.fit(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[i])
    Ys_preds_training = lr.predict(Xs_training)
    all_predicted_Ys_from_Xs_trainingSet.append(Ys_preds_training)
    Ys_preds_test = lr.predict(Xs_test)
    all_predicted_Ys_from_Xs_testSet.append(Ys_preds_test)
    tse = calcLostFunc(all_Alphas_Ys_Training_to_Xs_trainingSet[i],Ys_preds_training) / N
    te = calcLostFunc(all_Alphas_Ys_test_to_Xs_testSet[i],Ys_preds_test) / N
    tableOfResults_q2_A_and_B.append({"alpha": alphas[i],"teta_0_cova" : lr.intercept_[0], "teta_1_cova": list(lr.coef_.tolist())[0], "TSE" : tse[0] ,"TE" : te[0]})


file = open("res_A_B.txt", "w")
for row in tableOfResults_q2_A_and_B:
    row_string = row.__str__()+"\n"
    file.write(row_string)


file.close()
print("DONE")


plt.axis([-6, 6, -6, 6])

plt.subplot(131)
plt.suptitle("0.1     0.3      0.5")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[0], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[0])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[0])
plt.subplot(132)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[1], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[1])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[1])
plt.subplot(133)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[2], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[2])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[2])
plt.show()

plt.suptitle("0.7     0.9      1.1")
plt.subplot(131)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[3], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[3])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[3])
plt.subplot(132)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[4], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[4])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[4])
plt.subplot(133)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[5], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[5])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[5])
plt.show()

plt.suptitle("1.3     1.5      1.7")
plt.subplot(131)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[6], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[6])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[6])
plt.subplot(132)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[7], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[7])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[7])
plt.subplot(133)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[8], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[8])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[8])
plt.show()

plt.suptitle("1.9         2.1")
plt.subplot(131)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[9], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[9])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[9])
plt.subplot(132)
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet[10], "ro")
plt.plot(Xs_training, all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice[10])
plt.plot(Xs_training, all_predicted_Ys_from_Xs_trainingSet[10])

plt.show()










#seif 3

Cs_cova= []
as_cova = []
for row in tableOfResults_q2_A_and_B:
    t1 = row["teta_1_cova"][0]
    t0 = row["teta_0_cova"]
    Cs_cova.append(t0)
    as_cova.append(t1/t0)

Cs_cova_Abs_diff_C = []
as_cova_Abs_diff_alphas = []

for c in Cs_cova:
    res = np.abs(c-C)
    Cs_cova_Abs_diff_C.append(res)

for i in range(len(alphas)):
    res = np.abs(as_cova[i]-alphas[i])
    as_cova_Abs_diff_alphas.append(res)

for i in range(len(alphas)):
    print("alpha : ",alphas[i] , "|C^ - C | = " ,Cs_cova_Abs_diff_C[i] ,"|a^ - a| = " , as_cova_Abs_diff_alphas[i] ,"\n")

plt.subplot(131)
plt.suptitle("lab 1 part 3 B")
plt.plot(alphas, as_cova_Abs_diff_alphas,'r')
plt.plot(alphas, Cs_cova_Abs_diff_C,'g')
plt.show()




