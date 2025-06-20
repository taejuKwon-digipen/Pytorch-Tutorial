import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features= 1, noise=20, random_state= 1)

#Numpy배열을 pytorch tensor로 변환
#astype(np.float32) 는 pytorch의 float 32 타입과 맞추기 위해 사용
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

#TODO
# 1) model
input_size = n_features #입력 1차원
output_size = 1 #출력 선형회귀
model = nn.Linear(input_size, output_size) #nn.Linear는 내부적으로 y = w*x + b ~> weight + bias포함된 모델

# 2) loss and optimizier
learning_rate = 0.01 #학습률
criterion = nn.MSELoss() #평균 제곱 오차 (MES) - 회귀에서 자주 씀
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #SGD사용, model.parameters()는 학습대상

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X) #순전파 : w,b로 y예측
    loss = criterion(y_predicted, y) #예측 vs 실제값 차이 측정

    #backward pass
    loss.backward() #모든 파라미터에 대해 기울기 .grad자동 계산 &&  MSB기준으로 미분

    #update
    optimizer.step() #.grad를 이용해서 실제 w, b를 갱신

    optimizer.zero_grad()
    

    if(epoch-1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy() #그래프 시각화
plt.plot(X_numpy, y_numpy, 'ro')     # 원래 데이터 (빨간 점)
plt.plot(X_numpy, predicted, 'b')    # 예측 직선 (파란 선)

plt.show()