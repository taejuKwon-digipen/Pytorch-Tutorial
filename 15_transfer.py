#Image Folder
#Scheduler -> to change learning rate
#Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib as plt
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array
std = np.array

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), #랜덤 크롭
        transforms.RandomHorizontalFlip(), #랜덤 수평 뒤집기
        transforms.ToTensor(), #tensor로 변환
        transforms.Normalize(mean, std) #이미지 정규화
    ]),
    'val' : transforms.Compose([ 
        transforms.Resize(256), #리사이즈
        transforms.CenterCrop(224), #중앙 크롭
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

#이미지 폴더 경로 정의
data_dir = 'data/hymenoptera_data'
#데이터 셋 생성 
sets= ['train', 'val']
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x])
                  for x in ['train', 'val']}
#dataloader 정의 (배치사이즈4, 무작위 셔플)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 4,
                                              shuffle=True, num_workers=0)
                for x in ['train', 'val']}
#각 데이터셋 크기
datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#클래스 이름 추출
class_names = image_datasets['train'].classes
print(class_names)


#학습 함수
def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time() 

    best_model_wts = copy.deepcopy(model.state_dict()) #가장 성능 좋은 모델 저장
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() #학습모드
            else:
                model.eval() #평가모드

            running_loss=0.0
            running_corrects= 0

            for inputs, labels in dataloaders[phase]: # 배치단위 루프
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'): #train일때만 gradient계산
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 가장 확률높은 클래스 예측
                    loss = criterion(outputs, labels) #손실 계산

                    if(phase == 'train') :
                        optimizer.zero_grad() # 기존 gradient 초기화
                        loss.backward() # 역전파
                        optimizer.step() # optimizer 갱신

                running_loss += loss.item() * inputs.size(0) #배치열 손실 합산
                running_corrects += torch.sum(preds == labels.data) #정답 수 합산

            if(phase == 'train'):
                scheduler.step() #스케쥴러로 learning rate 조정

            epoch_loss = running_loss / datasets_sizes[phase] # epoch 평균 손실
            epoch_acc = running_corrects.double() / datasets_sizes[phase] # 정확도

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # 정확도가 더 좋아지면 best 모델로 저장
            if(phase == 'val' and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts) # 가장 좋은 가중치 로드
    return model


#tranfer

#미리 학습된 ResNet18 모델 불러오기
model= models.resnet18(pretrained = True)
# ResNet의 마지막 fully connected layer의 입력 수 가져오기
num_ftrs = models.fc.in_features #fully connected

# 마지막 fc 레이어를 클래스 수(2개)로 교체 (이진 분류)
model.fc = nn.Linear(num_ftrs, 2)
# 모델을 GPU 또는 CPU에 할당
model.to(device)

# 손실함수로 CrossEntropyLoss 사용 (다중 클래스 분류용)
criterion = nn.CrossEntropyLoss()
# 옵티마이저는 SGD (확률적 경사하강법)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

#scheduler
# 학습률 스케줄러: 7 epoch마다 lr을 10%로 감소
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1) #every 7 epochs learning rate multiplied with 0.1 (only updated 10 percent)
# 모델 학습 시작
model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)

#진행 순서
#1. ImageFolder를 기반으로 이미지 불러옴
#2. 전처리를 적용
#3. pretrained ResNet18 사용 -> Transfer Learning 구조
#4. 2개의 클래스 분류기 학습
#5. 학습률 스케쥴러로 학습 속도 조절
#6. 최종적으로 가장 정확도 높은 모델 반환
