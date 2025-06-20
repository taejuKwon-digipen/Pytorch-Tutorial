import numpy as np

#데이터 정의
X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype =np.float32)

#모델 가중치 초기화
w = 0.0

#model prediction
def forward(x) :
    return w * x

#loss = MSE (mean squared error) -> 손실함수
def loss(y, y_predicted):
    #gradient
    return ((y_predicted - y)**2).mean()

#gradient
#MSE = 1/N = (w * x - y)**2
#dj/dw = 1/N 2x (w+x -y)

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted -y).mean()

print(f'Prediction before training: f(5)= {forward(5):.3f}')

#Training
learning_rate = 0.01 #가중치를 얼마나 크게 바꿀지 결정하는 하이퍼파라미터
n_iters = 10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X, Y, y_pred)

    #update weights
    w -=learning_rate*dw

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w: .3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')