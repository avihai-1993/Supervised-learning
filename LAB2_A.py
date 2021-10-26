

import numpy as np
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

TASK_SIGMA = 5
N = 250

def calcLostFunc(y1 ,y2):
    res = []
    if len(y1) == len(y2):
        for i in range(len(y1)):
            res.append((y1[i]-y2[i])*(y1[i]-y2[i]))
            

    return sum(res)

def splitData(start_include, end, X):
    if type(X) is not list:
        return None, None

    x_split = X[int(len(Xs) * start_include):int(len(Xs) * end)]
    x_rest = X[0: int(len(Xs) * start_include)] + X[int(len(Xs) * end):len(Xs)]

    return x_split, x_rest



def task_func(x,sigma):
    epsilon = np.random.normal(0,sigma)
    return -3+1.6*x-0.3*x**2+x**3-0.1*x**4+epsilon

def caculate_ys(X,sigma):
    y = []
    for x in X:
        y.append(task_func(x,sigma))

    return y

def set_of_power(x,p):
    return [x**(i+1) for i in range(p)]

def x17_set(X):
    res_x = []
    for x in X:
        res_x.append(set_of_power(x,17))

    return res_x

# q A
def LAB2_A(x17_training,y,r):

    rr = Ridge(r)
    rr.fit(x17_training,y)
    Xs_test = np.linspace(-2.5, 2.5, 10 ** 5)
    Xs17_test = x17_set(Xs_test)
    Ys_test = caculate_ys(Xs_test,TASK_SIGMA)
    pred = rr.predict(Xs17_test)
    te = TASK_SIGMA**2 + (1/5) * calcLostFunc(pred,Ys_test)
    return te

#q B
def LAB2_Task_2_helper(r, X, Ys, sigma, start_include, end, toTest=False):
    if r < 0:
        return None
    if toTest:
        X17_test, X17_training = splitData(start_include, end, x17_set(X))
        Ys_test,Ys_training = splitData(start_include, end, Ys)
    else:
        X17_training, X17_test = splitData(start_include, end, x17_set(X))
        Ys_training, Ys_test = splitData(start_include, end, Ys)
    rr = Ridge(r)
    rr.fit(X17_training,Ys_training)
    pred = rr.predict(X17_test)
    return sigma**2 + (1/5) * calcLostFunc(pred,Ys_test)

#q B
def LAB2_Task_2(r,X,Y,sigma):
    mse = []
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0, 0.1, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.1, 0.2, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.2, 0.3, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.3, 0.4, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.4, 0.5, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.5, 0.6, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.6, 0.7, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.7, 0.8, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.8, 0.9, True))
    mse.append(LAB2_Task_2_helper(r, X, Y, sigma, 0.9, 1, True))
    return np.average(mse)

#q 3
def LAB2_task_3(rss,X,Y,sigma):
    Tes = []
    MES_s = []

    for r in rss:
        #TE
        Tes.append(LAB2_A(x17_set(X),Y,r))
        #MES
        MES_s.append(LAB2_Task_2(r, X, Y, sigma))


    return Tes, MES_s, rss[np.argmin(Tes)],rss[np.argmin(MES_s)]


r = 0.12 # that could be positive random number
Xs = np.random.uniform(-2.5, 2.5, N)
Ys = caculate_ys(Xs,TASK_SIGMA)
Xs17 = x17_set(Xs)



# must give x17
a = LAB2_A(Xs17,Ys,r)
print(a)

#make x x17 automatic inside the function
b = LAB2_Task_2(r,Xs,Ys,TASK_SIGMA)
print(b)



########################  part 3

#making the Rs
rs_start = 0.0001
rs_end = 20 #100
rs_step = 1 #0.2
rs_n = int((rs_end-rs_start)/rs_step)
rs = np.linspace(rs_start,rs_end,rs_n)
rs_log_invers=list(map(lambda r : -1*np.log(r),rs))



# exec the task
T, M, r_of_te_min, r_of_mes_min = LAB2_task_3(rs,Xs,Ys,TASK_SIGMA)


print("min te :  ", r_of_te_min)
print("min mes : ", r_of_mes_min)

plt.axis([-200, 200, -200, 200])
plt.subplot(131)
plt.suptitle("TE    MES")
plt.plot(rs_log_invers,T)
plt.subplot(132)
plt.plot(rs_log_invers,M)
plt.show()


 ############## part 4 q 4

#a
R_TE_MINS = []
#b
R_TECV_MINS = []
#c
R_TELSQ = []

r_MLSQ = 0.00001
MDGAM = 20 #200

for i in range(MDGAM):
    print("start ",i)
    X_s = np.random.uniform(-2.5, 2.5, N)
    Y_s = caculate_ys(Xs, TASK_SIGMA)
    no_need_one , no_need_two, r_te_min, r_mse_min = LAB2_task_3(rs, X_s, Y_s, TASK_SIGMA)
    R_TE_MINS.append(r_te_min)
    resForB = LAB2_A(x17_set(X_s),Y_s,r_mse_min)
    R_TECV_MINS.append(resForB)
    resForC = LAB2_A(x17_set(X_s),Y_s,r_MLSQ)
    R_TELSQ.append(resForC)
    print("finish ",i)


n_bins = 55
fig, axs = plt.subplots(1, 3)
axs[0].hist(R_TE_MINS ,bins=n_bins)
axs[1].hist(R_TECV_MINS,bins=n_bins)
axs[2].hist(R_TELSQ, bins=n_bins)
plt.show()
