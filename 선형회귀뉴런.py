#산점도 준비
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:,2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#뉴런 클래스 만들기
class Neuron:
  def __init__(self):
    self.w=1.0
    self.b=1.0

  #정방향 계산
  def forpass(self,x):
    y_hat=x*self.w+self.b
    return y_hat

  #역방향 계산
  def backprop(self,x,err):
    w_grad=x*err
    b_grad=1*err
    return w_grad, b_grad

  #훈련을 위한 fit() 매서드 구현
  def fit(self,x,y,epochs=100):
    for i in range(epochs):
      for x_i, y_i in zip(x,y):
        y_hat=self.forpass(x_i)
        err=-(y_i-y_hat)
        w_grad, b_grad=self.backprop(x_i,err)
        self.w -= w_grad #가중치 업데이트
        self.b -= b_grad #절편 업데이트

neuron = Neuron()
neuron.fit(diabetes.data[:,2], diabetes.target)

plt.scatter(diabetes.data[:,2], diabetes.target)
pt1 = (-0.1, -0.1*neuron.w + neuron.b)
pt2 = (0.15,0.15*neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
