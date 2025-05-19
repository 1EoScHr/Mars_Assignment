import re
import matplotlib.pyplot as plt

log_text = """
Epoch 1/200: 184it [00:21,  8.75it/s, backboneFreezed=1, trainLossPerBatch=8.57]                                                  
Epoch 1/200: 26it [00:02, 12.31it/s, validationLossPerBatch=5.61]                                   
[2025-05-20T02:28:46.103393|base.py:165] Caching best weights at epoch 1...
Epoch 2/200: 184it [00:20,  8.78it/s, backboneFreezed=1, trainLossPerBatch=6.47]                                                  
Epoch 2/200: 26it [00:02, 10.43it/s, validationLossPerBatch=5.53]                                   
[2025-05-20T02:29:09.623921|base.py:165] Caching best weights at epoch 2...
Epoch 3/200: 184it [00:20,  9.09it/s, backboneFreezed=1, trainLossPerBatch=5.68]                                                  
Epoch 3/200: 26it [00:02, 12.22it/s, validationLossPerBatch=5.47]                                   
[2025-05-20T02:29:32.069563|base.py:165] Caching best weights at epoch 3...
Epoch 4/200: 184it [00:29,  6.13it/s, backboneFreezed=0, trainLossPerBatch=5.17]                                                  
Epoch 4/200: 26it [00:02, 11.76it/s, validationLossPerBatch=5.86]                                   
Epoch 5/200: 184it [00:30,  6.12it/s, backboneFreezed=0, trainLossPerBatch=5.29]                                                  
Epoch 5/200: 26it [00:02, 11.50it/s, validationLossPerBatch=5.73]                                   
Epoch 6/200: 184it [00:30,  6.12it/s, backboneFreezed=0, trainLossPerBatch=5.54]                                                  
Epoch 6/200: 26it [00:02, 11.44it/s, validationLossPerBatch=5.05]                                   
[2025-05-20T02:31:09.114766|base.py:165] Caching best weights at epoch 6...
Epoch 7/200: 184it [00:29,  6.22it/s, backboneFreezed=0, trainLossPerBatch=5.36]                                                  
Epoch 7/200: 26it [00:02, 11.72it/s, validationLossPerBatch=5.7]                                    
Epoch 8/200: 184it [00:31,  5.89it/s, backboneFreezed=0, trainLossPerBatch=5.38]                                                  
Epoch 8/200: 26it [00:02, 10.78it/s, validationLossPerBatch=5.73]                                   
Epoch 9/200: 184it [00:31,  5.87it/s, backboneFreezed=0, trainLossPerBatch=5.48]                                                  
Epoch 9/200: 26it [00:02, 12.14it/s, validationLossPerBatch=5.21]                                   
Epoch 10/200: 184it [00:30,  6.04it/s, backboneFreezed=0, trainLossPerBatch=5.23]                                                 
Epoch 10/200: 26it [00:02, 11.59it/s, validationLossPerBatch=6.44]                                  
Epoch 11/200: 184it [00:29,  6.18it/s, backboneFreezed=0, trainLossPerBatch=5.83]                                                 
Epoch 11/200: 26it [00:02, 11.93it/s, validationLossPerBatch=5.39]                                  
Epoch 12/200: 184it [00:31,  5.79it/s, backboneFreezed=0, trainLossPerBatch=5.28]                                                 
Epoch 12/200: 26it [00:02, 10.83it/s, validationLossPerBatch=5.6]                                   
Epoch 13/200: 184it [00:30,  5.94it/s, backboneFreezed=0, trainLossPerBatch=5.31]                                                 
Epoch 13/200: 26it [00:02, 11.90it/s, validationLossPerBatch=4.22]                                  
[2025-05-20T02:35:00.448890|base.py:165] Caching best weights at epoch 13...
Epoch 14/200: 184it [00:31,  5.82it/s, backboneFreezed=0, trainLossPerBatch=4.86]                                                 
Epoch 14/200: 26it [00:02, 12.29it/s, validationLossPerBatch=5.22]                                  
Epoch 15/200: 184it [00:31,  5.92it/s, backboneFreezed=0, trainLossPerBatch=4.94]                                                 
Epoch 15/200: 26it [00:02, 11.53it/s, validationLossPerBatch=5.17]                                  
Epoch 16/200: 184it [00:30,  6.00it/s, backboneFreezed=0, trainLossPerBatch=5.41]                                                 
Epoch 16/200: 26it [00:02, 11.26it/s, validationLossPerBatch=5.61]                                  
Epoch 17/200: 184it [00:30,  6.04it/s, backboneFreezed=0, trainLossPerBatch=5.28]                                                 
Epoch 17/200: 26it [00:02, 11.96it/s, validationLossPerBatch=5.42]                                  
Epoch 18/200: 184it [00:30,  6.04it/s, backboneFreezed=0, trainLossPerBatch=5.19]                                                 
Epoch 18/200: 26it [00:02, 11.37it/s, validationLossPerBatch=4.15]                                  
[2025-05-20T02:37:46.291784|base.py:165] Caching best weights at epoch 18...
Epoch 19/200: 184it [00:30,  6.04it/s, backboneFreezed=0, trainLossPerBatch=5.16]                                                 
Epoch 19/200: 26it [00:02, 10.25it/s, validationLossPerBatch=5.77]                                  
Epoch 20/200: 184it [00:30,  6.04it/s, backboneFreezed=0, trainLossPerBatch=4.74]                                                 
Epoch 20/200: 26it [00:02, 11.12it/s, validationLossPerBatch=4.51]                                  
Epoch 21/200: 184it [00:30,  6.01it/s, backboneFreezed=0, trainLossPerBatch=5.03]                                                 
Epoch 21/200: 26it [00:02, 11.32it/s, validationLossPerBatch=4.91]                                  
Epoch 22/200: 184it [00:30,  5.94it/s, backboneFreezed=0, trainLossPerBatch=5.39]                                                 
Epoch 22/200: 26it [00:02,  8.70it/s, validationLossPerBatch=4.5]                                   
Epoch 23/200: 184it [00:30,  5.97it/s, backboneFreezed=0, trainLossPerBatch=5.2]                                                  
Epoch 23/200: 26it [00:02, 11.10it/s, validationLossPerBatch=5.08]                                  
Epoch 24/200: 184it [00:31,  5.83it/s, backboneFreezed=0, trainLossPerBatch=5.56]                                                 
Epoch 24/200: 26it [00:02, 10.00it/s, validationLossPerBatch=4.49]                                  
Epoch 25/200: 184it [00:29,  6.15it/s, backboneFreezed=0, trainLossPerBatch=5.35]                                                 
Epoch 25/200: 26it [00:02, 11.48it/s, validationLossPerBatch=5.75]                                  
Epoch 26/200: 184it [00:29,  6.19it/s, backboneFreezed=0, trainLossPerBatch=5.5]                                                  
Epoch 26/200: 26it [00:02, 12.32it/s, validationLossPerBatch=5.13]                                  
Epoch 27/200: 184it [00:29,  6.20it/s, backboneFreezed=0, trainLossPerBatch=5.17]                                                 
Epoch 27/200: 26it [00:02, 11.80it/s, validationLossPerBatch=5.71]                                  
Epoch 28/200: 184it [00:29,  6.17it/s, backboneFreezed=0, trainLossPerBatch=5.02]                                                 
Epoch 28/200: 26it [00:02, 11.34it/s, validationLossPerBatch=5.86]                                  
Epoch 29/200: 184it [00:29,  6.14it/s, backboneFreezed=0, trainLossPerBatch=5.47]                                                 
Epoch 29/200: 26it [00:02, 10.98it/s, validationLossPerBatch=4.86]                                  
Epoch 30/200: 184it [00:29,  6.19it/s, backboneFreezed=0, trainLossPerBatch=5.07]                                                 
Epoch 30/200: 26it [00:02, 12.02it/s, validationLossPerBatch=5.69]                    
"""

# 正则表达式提取epoch编号和损失
train_pattern = re.compile(r"Epoch (\d+)/\d+: \d+it \[.*trainLossPerBatch=([\d\.]+)\]")
val_pattern = re.compile(r"Epoch (\d+)/\d+: \d+it \[.*validationLossPerBatch=([\d\.]+)\]")

train_losses = {}
val_losses = {}

for line in log_text.splitlines():
    train_match = train_pattern.search(line)
    if train_match:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        train_losses[epoch] = loss
    val_match = val_pattern.search(line)
    if val_match:
        epoch = int(val_match.group(1))
        loss = float(val_match.group(2))
        val_losses[epoch] = loss

# 按epoch排序，准备画图数据
epochs = sorted(set(train_losses.keys()) & set(val_losses.keys()))
train_loss_list = [train_losses[e] for e in epochs]
val_loss_list = [val_losses[e] for e in epochs]

# 画折线图
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss_list, label="Train Loss", marker='o')
plt.plot(epochs, val_loss_list, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
print("Loss curve saved as loss_curve.png")