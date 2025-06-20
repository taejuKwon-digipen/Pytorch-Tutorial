# 1) Design model(input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn

#데이터 정의
X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32) #이차원 배열이어야 함
Y = torch.tensor([[2], [4], [6], [8]], dtype =torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)
print(f'Prediction before training: f(5)= {model(X_test).item():.3f}')

#Training
learning_rate = 0.01 #가중치를 얼마나 크게 바꿀지 결정하는 하이퍼파라미터
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #경사하강법 optim

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward() #dl/dx

    #update weights
    optimizer.step()

    #zero Grdients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() #b = bias
        print(f'epoch {epoch + 1}: w = {w[0][0].item(): .3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')