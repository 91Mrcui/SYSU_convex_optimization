import numpy as np
import matplotlib.pyplot as plt
import mnist

#Data: read MNIST
DATA_PATH = './mnist'
mn = mnist.MNIST(DATA_PATH)
mn.gz = True # Enable loading of gzip-ed files
train_img, train_label = mn.load_training()
test_img, test_label = mn.load_testing()
train_X = np.array(train_img)
train_Y = np.array(train_label)
test_X = np.array(test_img)
test_Y = np.array(test_label)

num_classes = 10
train_Y_len = len(train_Y)


#Hyperparameters
lr = 0.00000001
epochs = 50
lamda = 0.001

#Model
W = np.zeros((784, num_classes))
W_y = np.zeros((784, num_classes))
W_z = np.zeros((784, num_classes))

loss_record=[]
for step in range(epochs):
  pred = train_X.dot(W)

  # 补充完整的代码
  #------------------------------
  N = train_Y_len
  dw = np.zeros(W.shape)
  loss = 0.0

  score = train_X.dot(W) #caculate score
  score_y = score[range(N),train_Y]
  score_y=score_y.reshape(-1,1)
  diff = score - score_y + 1 
  diff[range(N),train_Y] =0 # j!=yi,if j=yi loss=0

  tmp=np.zeros(diff.shape)
  loss = np.sum(np.maximum(diff,tmp)*np.maximum(diff,tmp)) / N+lamda*np.sum(W*W)
  loss_record.append(loss)
  d_loss=2*np.sum(np.maximum(diff,tmp))/N # 求导的一部分

  s = diff>0   #if score>0,true
  num_of_score = np.sum(s,axis=1)  # 每个数据得分大于0的个数
  M = np.zeros(s.shape)
  M[range(N),train_Y] = -num_of_score  #j=yi，dLi/dw_yi =Xi*mark
  M = M + s #diff>0 and j!=yi ,mark=1

  dw = d_loss * train_X.T.dot(M)/N + 2*lamda*W 
  W_y=W-lr*dw
  W_z=W_z-(float(epochs+1)*lr/2)*dw
  W=(float(epochs+1)/float(epochs+3))*W_y+(2/float(epochs+3))*W_z

  #------------------------------

  pred = test_X.dot(W)
  pred = np.argmax(pred, 1)

  acc = np.equal(test_Y, pred).sum() / len(test_X)
  print("[%d/%d]LOSS:%.3f, ACC:%.3f" %(step+1, epochs, loss, acc))
  
plt.plot(loss_record)
plt.title("APGD")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()