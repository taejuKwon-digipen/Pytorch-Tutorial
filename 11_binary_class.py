import torch
import torch.nn as nn

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        print("After Linear:", x)
        out = self.relu(out)
        print("After ReLU:", out)
        out = self.linear2(out)
        print("After linear2:", out)

        #sigmoid at the end
        y_pred = torch.sigmoid(out)
        print("🔹 After Sigmoid (Final Output):", y_pred)
        return y_pred

model = NeuralNet1(input_size = 28*28, hidden_size= 5)
criterion = nn.BCELoss()
