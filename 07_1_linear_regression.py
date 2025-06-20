# Pytorch 로 유방암 데이터를 분류하는 로지스틱 회귀 분류기

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer() # 유방암 데이터 셋 로드
X,y = bc.data, bc.target # X입력값, y출력값(0 = 악성, 1 = 양성)

n_samples, n_features = X.shape

#전체 데이터를 80:20으로 나눔
#random_state 고정으로 결과 재현 가능
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1)model
# f = wx + b, sigmoid at the end

#모델 정의
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) #sigmoid : 0->1확률로 출력
        return y_predicted

model = LogisticRegression(n_features) #n_features=30으로 지정해서 모델 인스턴스 생성

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() #BCELoss : Binary Cross Entrophy 확률값과 타겟값 차이 측정
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
num_epochs = 100 #전체 데이터를 100번 학습 
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    #backward pass
    loss.backward() # 기울기 계산

    optimizer.step() # 파라미터(w, b) 업데이트

    #updates
    optimizer.zero_grad() # 기울기 초기화

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#평가 (inference)
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() #예측 확률이 0.5 이상이면 1, 아니면 0 → 분류 결과
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #예측값과 실제값 비교->정확도계산
    print(f'accuracy = {acc:.4f}')