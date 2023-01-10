import numpy as np
import matplotlib.pyplot as plt
nSample = 100

#Data
X = np.linspace(-1, 1, nSample)
GaussianNoise = np.random.normal(0, 1, nSample)
Y = 10 * X + GaussianNoise

#Model
W = np.array([0])

#Loss Function
loss = ((Y - W*X) **2).mean()
print("LOSS:%.3f" %(loss))

lr = 0.1
epochs = 1000

#change_loss=[] #Record the change of loss function
#pre_loss=loss  #Record the last loss function
for step in range(epochs):
    pred = W * X
	#Optimization：gradient descend
    grad = 2*(pred-Y)*X
    W = W-lr*grad
    loss = ((Y - pred) **2).mean()
    #if loss>=pre_loss:
    #    lr*=0.9
    #change_loss.append(loss)
    #pre_loss=loss
    print("[%d/%d]LOSS:%.3f" %(step + 1, epochs, loss))

pred = W * X
plt.plot(X, Y, color = 'red', label= 'GroundTruth')
plt.plot(X, pred, color = 'blue', label= 'Prediction')
plt.legend()
plt.title('Results after training for '+str(epochs)+' epochs')
plt.savefig('Results.png')
plt.show()
'''
plt.plot(range(epochs),change_loss)
plt.xlabel('literaton')
plt.ylabel('loss')
plt.show()
'''