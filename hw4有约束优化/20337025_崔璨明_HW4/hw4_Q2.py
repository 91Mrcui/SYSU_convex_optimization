import numpy as np
import time
 
#计算梯度
def grad(t, x1, x2):
    return np.array([[t * (4*x1 + 2*x2 - 10) + (2*x1)/(1 - pow(x1,2) - pow(x2,2)) ],
                     [t * (2*x2 + 2*x1 - 10) + (2*x2)/(1 - pow(x1,2) - pow(x2,2)) ]])
 
#计算Hessian矩阵
def Hessian(t, x1, x2):
    return np.array([[4*t + ( 2*(1-pow(x1,2)-pow(x2,2)) + 4*pow(x1,2) ) / pow((1-pow(x1,2)-pow(x2,2)), 2) , 2*t + (4*x1*x2) / pow((1-pow(x1,2)-pow(x2,2)), 2)],

                     [2*t + (4*x1*x2) / pow((1-pow(x1,2)-pow(x2,2)), 2) , 2*t + (2*(1-pow(x1,2)-pow(x2,2)) + 4*pow(x2,2) ) / pow((1-pow(x1,2)-pow(x2,2)), 2) ]])
 
 
def Newton_Raphson(t, x1, x2):
    gf = grad(t, x1, x2)
 
    Hf = Hessian(t, x1, x2)
    Hf_inv = np.linalg.inv(Hf)
 
    deltaX = 0.1 * np.matmul(Hf_inv, gf)
    res = np.linalg.norm(deltaX, 2)
 
    return x1 - deltaX[0, 0], x2 - deltaX[1, 0], res
 
 
if __name__ == "__main__":
    time_start = time.time()
    t = 2
    x1 = 0.3
    x2 = 0.4
    while True:
        while True:
            x1, x2, res = Newton_Raphson(t, x1, x2)
            if res < 0.0001:
                break

        if 3.0 / t < 0.0001:
            time_end = time.time()
            print('Result:\ncosting time:', time_end - time_start)
            print("t:{}\nx1:{}\nx2:{}".format(t, x1, x2))
            break
        t = 2 * t

    print("mini value:{}".format(2*pow(x1,2)+2*x1*x2+pow(x2,2)-10*x1-10*x2))