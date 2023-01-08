import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 2),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(2, input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def fit(self, dataloader, epoch_num=10):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # 訓練模型
        for epoch in range(epoch_num):  # 訓練 10 個 epoch
            for data in dataloader:  # 取出一批資料
                x, _ = data 
                encoded, decoded = self.forward(x)  # 進行預測

                loss = criterion(decoded, x)  # 計算損失
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向傳播
                optimizer.step()  # 更新參數
                
            print("Epoch: {}/10, Loss: {:.4f}".format(epoch+1, loss.item()))  # 輸出訓練結果