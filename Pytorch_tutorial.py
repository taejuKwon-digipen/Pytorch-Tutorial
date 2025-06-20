# import torch

# weights = torch.ones(4, requires_grad = True) 
# # 1이 4개짜리 벡터
# # requires_grad = True :이 텐서에 대해 gradient추적하겠음.
# # weights = 변수 (모델에서 학습해야 할 파라미터로 자주 쓰임)

# for epoch in range(1): #에폭 학습단계
#     model_output = (weights * 3).sum() # 벡터 곱셈 + 더하기 = 12 -> autograd가 연산과정을 기억
#     model_output.backward() # model_output(12) 리준으로 weight에 대한 grad를 자동 계산
#     # 즉, model_output = f(weights)일때 df/dweight 자동 계산
#     print(weights.grad)
#     weights.grad.zero_()


# import torch

# weights = torch.ones(4, requires_grad= True)
# z.backward()
# weights.grad.zero_()
#--------------------------------------------------------------------------------------------#
#선형회귀
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad= True) #가중치

#~왜 grad를 계산하느냐
# 모델의 loss를 줄이려면 W(가중치)를 요리조리 바꿔야되는데 어떻게 바꿀지를 grad를 보고 판단함

# forward pass and compute the loss
y_hat = w*x 
loss = (y_hat - y) **2
print (loss)

# backward pass
loss.backward() #whole gradient computation
print(w.grad)

