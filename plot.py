import re
import matplotlib.pyplot as plt

log_text = """                                                                                                
[2025-05-22T21:15:10.708410|base.py:165] Caching best weights at epoch 10...
Epoch 11/20: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=5.51]                                                  
Epoch 11/20: 26it [00:02, 11.38it/s, validationLossPerBatch=4.87]                                   
[2025-05-22T21:15:47.851313|base.py:165] Caching best weights at epoch 11...
Epoch 12/20: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=5.39]                                                  
Epoch 12/20: 26it [00:02, 11.06it/s, validationLossPerBatch=5.1]                                    
Epoch 13/20: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=5.26]                                                  
Epoch 13/20: 26it [00:02, 11.05it/s, validationLossPerBatch=4.65]                                   
[2025-05-22T21:17:02.281421|base.py:165] Caching best weights at epoch 13...
Epoch 14/20: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=5.17]                                                  
Epoch 14/20: 26it [00:02, 11.10it/s, validationLossPerBatch=4.65]                                   
Epoch 15/20: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=5.06]                                                  
Epoch 15/20: 26it [00:02, 10.82it/s, validationLossPerBatch=4.49]                                   
[2025-05-22T21:18:16.905862|base.py:165] Caching best weights at epoch 15...
Epoch 16/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.98]                                                  
Epoch 16/20: 26it [00:02, 10.38it/s, validationLossPerBatch=4.39]                                   
[2025-05-22T21:18:54.322299|base.py:165] Caching best weights at epoch 16...
Epoch 17/20: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=4.96]                                                  
Epoch 17/20: 26it [00:02, 10.38it/s, validationLossPerBatch=4.29]                                   
[2025-05-22T21:19:31.799239|base.py:165] Caching best weights at epoch 17...
Epoch 18/20: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.88]                                                  
Epoch 18/20: 26it [00:02, 11.19it/s, validationLossPerBatch=4.29]                                   
Epoch 19/20: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.87]                                                  
Epoch 19/20: 26it [00:02, 11.09it/s, validationLossPerBatch=4.26]                                   
[2025-05-22T21:20:46.054480|base.py:165] Caching best weights at epoch 19...
Epoch 20/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.83]                                                  
Epoch 20/20: 26it [00:02, 11.43it/s, validationLossPerBatch=4.24]          
Epoch 21/50: 184it [00:36,  5.04it/s, backboneFreezed=0, trainLossPerBatch=5.74]                                                  
Epoch 21/50: 26it [00:02, 11.04it/s, validationLossPerBatch=6.21]                                   
Epoch 22/50: 184it [00:36,  5.10it/s, backboneFreezed=0, trainLossPerBatch=5.46]                                                  
Epoch 22/50: 26it [00:02, 10.92it/s, validationLossPerBatch=5.29]                                   
Epoch 23/50: 184it [00:36,  5.07it/s, backboneFreezed=0, trainLossPerBatch=5.18]                                                  
Epoch 23/50: 26it [00:02, 11.70it/s, validationLossPerBatch=5.06]                                   
Epoch 24/50: 184it [00:35,  5.16it/s, backboneFreezed=0, trainLossPerBatch=4.98]                                                  
Epoch 24/50: 26it [00:02, 11.56it/s, validationLossPerBatch=4.83]                                   
Epoch 25/50: 184it [00:36,  5.09it/s, backboneFreezed=0, trainLossPerBatch=4.88]                                                  
Epoch 25/50: 26it [00:02, 10.71it/s, validationLossPerBatch=4.69]                                   
Epoch 26/50: 184it [00:35,  5.13it/s, backboneFreezed=0, trainLossPerBatch=4.74]                                                  
Epoch 26/50: 26it [00:02, 11.15it/s, validationLossPerBatch=4.44]                                   
Epoch 27/50: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=4.66]                                                  
Epoch 27/50: 26it [00:02, 10.97it/s, validationLossPerBatch=4.33]                                   
Epoch 28/50: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=4.57]                                                  
Epoch 28/50: 26it [00:02, 11.35it/s, validationLossPerBatch=4.3]                                    
Epoch 29/50: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=4.42]                                                  
Epoch 29/50: 26it [00:02, 10.98it/s, validationLossPerBatch=4.08]                                   
[2025-05-22T21:40:50.615528|base.py:165] Caching best weights at epoch 29...
Epoch 30/50: 184it [00:36,  5.10it/s, backboneFreezed=0, trainLossPerBatch=4.36]                                                  
Epoch 30/50: 26it [00:02, 10.69it/s, validationLossPerBatch=3.99]                                   
[2025-05-22T21:41:29.225953|base.py:165] Caching best weights at epoch 30...
Epoch 31/50: 184it [00:35,  5.21it/s, backboneFreezed=0, trainLossPerBatch=4.26]                                                  
Epoch 31/50: 26it [00:02, 10.91it/s, validationLossPerBatch=3.92]                                   
[2025-05-22T21:42:07.020311|base.py:165] Caching best weights at epoch 31...
Epoch 32/50: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=4.2]                                                   
Epoch 32/50: 26it [00:02, 10.97it/s, validationLossPerBatch=3.7]                                    
[2025-05-22T21:42:44.671287|base.py:165] Caching best weights at epoch 32...
Epoch 33/50: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.13]                                                  
Epoch 33/50: 26it [00:02, 10.68it/s, validationLossPerBatch=3.7]                                    
[2025-05-22T21:43:21.897379|base.py:165] Caching best weights at epoch 33...
Epoch 34/50: 184it [00:35,  5.17it/s, backboneFreezed=0, trainLossPerBatch=4.1]                                                   
Epoch 34/50: 26it [00:02, 11.63it/s, validationLossPerBatch=3.66]                                   
[2025-05-22T21:43:59.827739|base.py:165] Caching best weights at epoch 34...
Epoch 35/50: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=4]                                                     
Epoch 35/50: 26it [00:02, 11.37it/s, validationLossPerBatch=3.57]                                   
[2025-05-22T21:44:37.425680|base.py:165] Caching best weights at epoch 35...
Epoch 36/50: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=3.95]                                                  
Epoch 36/50: 26it [00:02, 11.46it/s, validationLossPerBatch=3.46]                                   
[2025-05-22T21:45:15.057057|base.py:165] Caching best weights at epoch 36...
Epoch 37/50: 184it [00:35,  5.14it/s, backboneFreezed=0, trainLossPerBatch=3.9]                                                   
Epoch 37/50: 26it [00:02, 11.06it/s, validationLossPerBatch=3.46]                                   
[2025-05-22T21:45:53.259555|base.py:165] Caching best weights at epoch 37...
Epoch 38/50: 184it [00:35,  5.16it/s, backboneFreezed=0, trainLossPerBatch=3.83]                                                  
Epoch 38/50: 26it [00:02, 10.99it/s, validationLossPerBatch=3.38]                                   
[2025-05-22T21:46:31.360673|base.py:165] Caching best weights at epoch 38...
Epoch 39/50: 184it [00:35,  5.18it/s, backboneFreezed=0, trainLossPerBatch=3.78]                                                  
Epoch 39/50: 26it [00:02, 10.74it/s, validationLossPerBatch=3.31]                                   
[2025-05-22T21:47:09.368820|base.py:165] Caching best weights at epoch 39...
Epoch 40/50: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=3.72]                                                  
Epoch 40/50: 26it [00:02, 10.33it/s, validationLossPerBatch=3.26]                                   
[2025-05-22T21:47:47.033225|base.py:165] Caching best weights at epoch 40...
Epoch 41/50: 184it [00:36,  5.02it/s, backboneFreezed=0, trainLossPerBatch=3.69]                                                  
Epoch 41/50: 26it [00:02, 10.46it/s, validationLossPerBatch=3.2]                                    
[2025-05-22T21:48:26.264157|base.py:165] Caching best weights at epoch 41...
Epoch 42/50: 184it [00:36,  5.07it/s, backboneFreezed=0, trainLossPerBatch=3.7]                                                   
Epoch 42/50: 26it [00:02, 11.17it/s, validationLossPerBatch=3.18]                                   
[2025-05-22T21:49:04.972608|base.py:165] Caching best weights at epoch 42...
Epoch 43/50: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=3.6]                                                   
Epoch 43/50: 26it [00:02, 11.09it/s, validationLossPerBatch=3.15]                                   
[2025-05-22T21:49:42.531182|base.py:165] Caching best weights at epoch 43...
Epoch 44/50: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=3.59]                                                  
Epoch 44/50: 26it [00:02, 10.47it/s, validationLossPerBatch=3.14]                                   
[2025-05-22T21:50:20.319016|base.py:165] Caching best weights at epoch 44...
Epoch 45/50: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.55]                                                  
Epoch 45/50: 26it [00:02, 11.19it/s, validationLossPerBatch=3.14]                                   
Epoch 46/50: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=3.54]                                                  
Epoch 46/50: 26it [00:02, 11.13it/s, validationLossPerBatch=3.1]                                    
[2025-05-22T21:51:35.174183|base.py:165] Caching best weights at epoch 46...
Epoch 47/50: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.52]                                                  
Epoch 47/50: 26it [00:02, 11.23it/s, validationLossPerBatch=3.09]                                   
[2025-05-22T21:52:12.403483|base.py:165] Caching best weights at epoch 47...
Epoch 48/50: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=3.52]                                                  
Epoch 48/50: 26it [00:02, 11.24it/s, validationLossPerBatch=3.1]                                    
Epoch 49/50: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.51]                                                  
Epoch 49/50: 26it [00:02, 11.31it/s, validationLossPerBatch=3.1]                                    
Epoch 50/50: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=3.52]                                                  
Epoch 50/50: 26it [00:02,  9.72it/s, validationLossPerBatch=3.08]  
Epoch 51/100: 184it [00:37,  4.94it/s, backboneFreezed=0, trainLossPerBatch=4.36]                                                 
Epoch 51/100: 26it [00:02, 10.47it/s, validationLossPerBatch=4.03]                                  
Epoch 52/100: 184it [00:36,  5.03it/s, backboneFreezed=0, trainLossPerBatch=4.33]                                                 
Epoch 52/100: 26it [00:02, 10.66it/s, validationLossPerBatch=4.05]                                  
Epoch 53/100: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=4.18]                                                 
Epoch 53/100: 26it [00:02, 11.12it/s, validationLossPerBatch=4.36]                                  
Epoch 54/100: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=4.04]                                                 
Epoch 54/100: 26it [00:02, 11.32it/s, validationLossPerBatch=3.82]                                  
Epoch 55/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.01]                                                 
Epoch 55/100: 26it [00:02, 11.25it/s, validationLossPerBatch=3.79]                                  
Epoch 56/100: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=3.93]                                                 
Epoch 56/100: 26it [00:02, 10.68it/s, validationLossPerBatch=3.64]                                  
Epoch 57/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.89]                                                 
Epoch 57/100: 26it [00:02, 11.12it/s, validationLossPerBatch=3.53]                                  
Epoch 58/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.82]                                                 
Epoch 58/100: 26it [00:02, 11.58it/s, validationLossPerBatch=3.41]                                  
Epoch 59/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.72]                                                 
Epoch 59/100: 26it [00:02, 11.33it/s, validationLossPerBatch=3.68]                                  
Epoch 60/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.71]                                                 
Epoch 60/100: 26it [00:02, 11.07it/s, validationLossPerBatch=3.45]                                  
Epoch 61/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.64]                                                 
Epoch 61/100: 26it [00:02, 11.31it/s, validationLossPerBatch=3.43]                                  
Epoch 62/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.62]                                                 
Epoch 62/100: 26it [00:02, 11.04it/s, validationLossPerBatch=3.26]                                  
Epoch 63/100: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=3.57]                                                 
Epoch 63/100: 26it [00:02, 10.15it/s, validationLossPerBatch=3.13]                                  
Epoch 64/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.56]                                                 
Epoch 64/100: 26it [00:02, 11.25it/s, validationLossPerBatch=3.14]                                  
Epoch 65/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.49]                                                 
Epoch 65/100: 26it [00:02, 11.58it/s, validationLossPerBatch=3.09]                                  
Epoch 66/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.45]                                                 
Epoch 66/100: 26it [00:02, 11.25it/s, validationLossPerBatch=3.02]                                  
[2025-05-22T22:12:00.519943|base.py:165] Caching best weights at epoch 66...
Epoch 67/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.43]                                                 
Epoch 67/100: 26it [00:02, 11.14it/s, validationLossPerBatch=3.68]                                  
Epoch 68/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.38]                                                 
Epoch 68/100: 26it [00:02, 11.41it/s, validationLossPerBatch=3.1]                                   
Epoch 69/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.36]                                                 
Epoch 69/100: 26it [00:02, 11.29it/s, validationLossPerBatch=3.06]                                  
Epoch 70/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.31]                                                 
Epoch 70/100: 26it [00:02, 10.96it/s, validationLossPerBatch=2.97]                                  
[2025-05-22T22:14:29.255902|base.py:165] Caching best weights at epoch 70...
Epoch 71/100: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=3.29]                                                 
Epoch 71/100: 26it [00:02, 10.91it/s, validationLossPerBatch=2.88]                                  
[2025-05-22T22:15:06.802862|base.py:165] Caching best weights at epoch 71...
Epoch 72/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.28]                                                 
Epoch 72/100: 26it [00:02, 11.31it/s, validationLossPerBatch=2.83]                                  
[2025-05-22T22:15:43.991470|base.py:165] Caching best weights at epoch 72...
Epoch 73/100: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=3.21]                                                 
Epoch 73/100: 26it [00:02, 11.72it/s, validationLossPerBatch=2.88]                                  
Epoch 74/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.16]                                                 
Epoch 74/100: 26it [00:02, 10.68it/s, validationLossPerBatch=2.8]                                   
[2025-05-22T22:16:58.557267|base.py:165] Caching best weights at epoch 74...
Epoch 75/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.15]                                                 
Epoch 75/100: 26it [00:02, 11.15it/s, validationLossPerBatch=2.77]                                  
[2025-05-22T22:17:35.788177|base.py:165] Caching best weights at epoch 75...
Epoch 76/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.14]                                                 
Epoch 76/100: 26it [00:02, 11.34it/s, validationLossPerBatch=2.76]                                  
[2025-05-22T22:18:12.906643|base.py:165] Caching best weights at epoch 76...
Epoch 77/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.09]                                                 
Epoch 77/100: 26it [00:02, 11.01it/s, validationLossPerBatch=2.88]                                  
Epoch 78/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.07]                                                 
Epoch 78/100: 26it [00:02, 11.12it/s, validationLossPerBatch=2.75]                                  
[2025-05-22T22:19:27.603610|base.py:165] Caching best weights at epoch 78...
Epoch 79/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.06]                                                 
Epoch 79/100: 26it [00:02, 11.47it/s, validationLossPerBatch=2.72]                                  
[2025-05-22T22:20:04.750098|base.py:165] Caching best weights at epoch 79...
Epoch 80/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.03]                                                 
Epoch 80/100: 26it [00:02, 11.46it/s, validationLossPerBatch=2.71]                                  
[2025-05-22T22:20:41.873279|base.py:165] Caching best weights at epoch 80...
Epoch 81/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.02]                                                 
Epoch 81/100: 26it [00:02, 11.23it/s, validationLossPerBatch=2.71]                                  
[2025-05-22T22:21:19.128079|base.py:165] Caching best weights at epoch 81...
Epoch 82/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3]                                                    
Epoch 82/100: 26it [00:02, 11.50it/s, validationLossPerBatch=2.66]                                  
[2025-05-22T22:21:56.203674|base.py:165] Caching best weights at epoch 82...
Epoch 83/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.99]                                                 
Epoch 83/100: 26it [00:02, 11.37it/s, validationLossPerBatch=2.63]                                  
[2025-05-22T22:22:33.328755|base.py:165] Caching best weights at epoch 83...
Epoch 84/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                 
Epoch 84/100: 26it [00:02, 11.17it/s, validationLossPerBatch=2.66]                                  
Epoch 85/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                 
Epoch 85/100: 26it [00:02, 11.18it/s, validationLossPerBatch=2.63]                                  
[2025-05-22T22:23:47.618602|base.py:165] Caching best weights at epoch 85...
Epoch 86/100: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                 
Epoch 86/100: 26it [00:02, 11.64it/s, validationLossPerBatch=2.59]                                  
[2025-05-22T22:24:24.896916|base.py:165] Caching best weights at epoch 86...
Epoch 87/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                 
Epoch 87/100: 26it [00:02, 11.75it/s, validationLossPerBatch=2.61]                                  
Epoch 88/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.92]                                                 
Epoch 88/100: 26it [00:02, 11.19it/s, validationLossPerBatch=2.61]                                  
Epoch 89/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                  
Epoch 89/100: 26it [00:02, 11.19it/s, validationLossPerBatch=2.6]                                   
Epoch 90/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                 
Epoch 90/100: 26it [00:02, 11.27it/s, validationLossPerBatch=2.59]                                  
[2025-05-22T22:26:53.748150|base.py:165] Caching best weights at epoch 90...
Epoch 91/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.89]                                                 
Epoch 91/100: 26it [00:02, 10.91it/s, validationLossPerBatch=2.56]                                  
[2025-05-22T22:27:30.974916|base.py:165] Caching best weights at epoch 91...
Epoch 92/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.85]                                                 
Epoch 92/100: 26it [00:02, 11.06it/s, validationLossPerBatch=2.59]                                  
Epoch 93/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                 
Epoch 93/100: 26it [00:02, 11.09it/s, validationLossPerBatch=2.56]                                  
[2025-05-22T22:28:45.337844|base.py:165] Caching best weights at epoch 93...
Epoch 94/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.84]                                                 
Epoch 94/100: 26it [00:02, 10.73it/s, validationLossPerBatch=2.56]                                  
[2025-05-22T22:29:22.790639|base.py:165] Caching best weights at epoch 94...
Epoch 95/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.85]                                                 
Epoch 95/100: 26it [00:02, 10.65it/s, validationLossPerBatch=2.56]                                  
Epoch 96/100: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.86]                                                 
Epoch 96/100: 26it [00:02, 11.45it/s, validationLossPerBatch=2.57]                                  
Epoch 97/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                 
Epoch 97/100: 26it [00:02, 10.95it/s, validationLossPerBatch=2.55]                                  
[2025-05-22T22:31:14.623960|base.py:165] Caching best weights at epoch 97...
Epoch 98/100: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.85]                                                 
Epoch 98/100: 26it [00:02, 11.17it/s, validationLossPerBatch=2.56]                                  
Epoch 99/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.84]                                                 
Epoch 99/100: 26it [00:02, 11.13it/s, validationLossPerBatch=2.57]                                  
Epoch 100/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.82]                                                
Epoch 100/100: 26it [00:02, 11.31it/s, validationLossPerBatch=2.58]    
Epoch 101/150: 184it [00:36,  5.05it/s, backboneFreezed=0, trainLossPerBatch=2.85]                                                
Epoch 101/150: 26it [00:02, 10.97it/s, validationLossPerBatch=2.62]                                 
Epoch 102/150: 184it [00:35,  5.18it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 102/150: 26it [00:02, 10.07it/s, validationLossPerBatch=2.56]                                 
Epoch 103/150: 184it [00:37,  4.96it/s, backboneFreezed=0, trainLossPerBatch=2.84]                                                
Epoch 103/150: 26it [00:02, 11.55it/s, validationLossPerBatch=2.58]                                 
Epoch 104/150: 184it [00:36,  5.10it/s, backboneFreezed=0, trainLossPerBatch=2.83]                                                
Epoch 104/150: 26it [00:02, 10.81it/s, validationLossPerBatch=2.54]                                 
[2025-05-22T22:43:02.910351|base.py:165] Caching best weights at epoch 104...
Epoch 105/150: 184it [00:35,  5.14it/s, backboneFreezed=0, trainLossPerBatch=2.83]                                                
Epoch 105/150: 26it [00:02, 11.02it/s, validationLossPerBatch=2.61]                                 
Epoch 106/150: 184it [00:36,  5.06it/s, backboneFreezed=0, trainLossPerBatch=2.82]                                                
Epoch 106/150: 26it [00:02, 10.81it/s, validationLossPerBatch=2.57]                                 
Epoch 107/150: 184it [00:35,  5.13it/s, backboneFreezed=0, trainLossPerBatch=2.81]                                                
Epoch 107/150: 26it [00:02, 10.37it/s, validationLossPerBatch=2.55]                                 
Epoch 108/150: 184it [00:37,  4.93it/s, backboneFreezed=0, trainLossPerBatch=2.81]                                                
Epoch 108/150: 26it [00:05,  4.40it/s, validationLossPerBatch=2.55]                                 
Epoch 109/150: 184it [00:37,  4.95it/s, backboneFreezed=0, trainLossPerBatch=2.77]                                                
Epoch 109/150: 26it [00:03,  7.20it/s, validationLossPerBatch=2.57]                                 
Epoch 110/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.79]                                                
Epoch 110/150: 26it [00:02, 11.32it/s, validationLossPerBatch=2.55]                                 
Epoch 111/150: 184it [00:35,  5.20it/s, backboneFreezed=0, trainLossPerBatch=2.77]                                                
Epoch 111/150: 26it [00:03,  8.57it/s, validationLossPerBatch=2.53]                                 
[2025-05-22T22:47:38.051018|base.py:165] Caching best weights at epoch 111...
Epoch 112/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.78]                                                
Epoch 112/150: 26it [00:02, 10.59it/s, validationLossPerBatch=2.55]                                 
Epoch 113/150: 184it [00:35,  5.18it/s, backboneFreezed=0, trainLossPerBatch=2.78]                                                
Epoch 113/150: 26it [00:02,  8.97it/s, validationLossPerBatch=2.55]                                 
Epoch 114/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.78]                                                
Epoch 114/150: 26it [00:02, 11.17it/s, validationLossPerBatch=2.55]                                 
Epoch 115/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.76]                                                
Epoch 115/150: 26it [00:02, 11.32it/s, validationLossPerBatch=2.54]                                 
Epoch 116/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.77]                                                
Epoch 116/150: 26it [00:03,  7.62it/s, validationLossPerBatch=2.54]                                 
Epoch 117/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.76]                                                
Epoch 117/150: 26it [00:02, 10.91it/s, validationLossPerBatch=2.53]                                 
[2025-05-22T22:51:23.751140|base.py:165] Caching best weights at epoch 117...
Epoch 118/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 118/150: 26it [00:02, 11.10it/s, validationLossPerBatch=2.53]                                 
[2025-05-22T22:52:01.161600|base.py:165] Caching best weights at epoch 118...
Epoch 119/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 119/150: 26it [00:02, 11.65it/s, validationLossPerBatch=2.52]                                 
[2025-05-22T22:52:38.272747|base.py:165] Caching best weights at epoch 119...
Epoch 120/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 120/150: 26it [00:02, 11.51it/s, validationLossPerBatch=2.52]                                 
Epoch 121/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 121/150: 26it [00:02, 11.29it/s, validationLossPerBatch=2.52]                                 
Epoch 122/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 122/150: 26it [00:02, 11.17it/s, validationLossPerBatch=2.55]                                 
Epoch 123/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 123/150: 26it [00:02, 11.41it/s, validationLossPerBatch=2.53]                                 
Epoch 124/150: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 124/150: 26it [00:02, 10.83it/s, validationLossPerBatch=2.53]                                 
Epoch 125/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 125/150: 26it [00:02, 11.50it/s, validationLossPerBatch=2.54]                                 
Epoch 126/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 126/150: 26it [00:02, 10.93it/s, validationLossPerBatch=2.53]                                 
Epoch 127/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 127/150: 26it [00:02, 10.62it/s, validationLossPerBatch=2.53]                                 
Epoch 128/150: 184it [00:35,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 128/150: 26it [00:02, 10.64it/s, validationLossPerBatch=2.53]                                 
Epoch 129/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 129/150: 26it [00:02,  8.77it/s, validationLossPerBatch=2.52]                                 
Epoch 130/150: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 130/150: 26it [00:02, 10.22it/s, validationLossPerBatch=2.51]                                 
[2025-05-22T22:59:29.336623|base.py:165] Caching best weights at epoch 130...
Epoch 131/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 131/150: 26it [00:02, 11.07it/s, validationLossPerBatch=2.54]                                 
Epoch 132/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 132/150: 26it [00:02, 11.00it/s, validationLossPerBatch=2.53]                                 
Epoch 133/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 133/150: 26it [00:02, 10.50it/s, validationLossPerBatch=2.51]                                 
[2025-05-22T23:01:21.178443|base.py:165] Caching best weights at epoch 133...
Epoch 134/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 134/150: 26it [00:02, 10.28it/s, validationLossPerBatch=2.54]                                 
Epoch 135/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 135/150: 26it [00:02, 10.99it/s, validationLossPerBatch=2.52]                                 
Epoch 136/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 136/150: 26it [00:02, 11.12it/s, validationLossPerBatch=2.51]                                 
[2025-05-22T23:03:13.341653|base.py:165] Caching best weights at epoch 136...
Epoch 137/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 137/150: 26it [00:02, 11.04it/s, validationLossPerBatch=2.53]                                 
Epoch 138/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 138/150: 26it [00:02, 10.76it/s, validationLossPerBatch=2.51]                                 
[2025-05-22T23:04:27.961112|base.py:165] Caching best weights at epoch 138...
Epoch 139/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 139/150: 26it [00:02, 10.46it/s, validationLossPerBatch=2.52]                                 
Epoch 140/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 140/150: 26it [00:02, 11.32it/s, validationLossPerBatch=2.52]                                 
Epoch 141/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.76]                                                
Epoch 141/150: 26it [00:02, 11.40it/s, validationLossPerBatch=2.51]                                 
Epoch 142/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 142/150: 26it [00:02, 10.96it/s, validationLossPerBatch=2.54]                                 
Epoch 143/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.76]                                                
Epoch 143/150: 26it [00:02, 11.50it/s, validationLossPerBatch=2.51]                                 
Epoch 144/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 144/150: 26it [00:02, 10.84it/s, validationLossPerBatch=2.51]                                 
Epoch 145/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 145/150: 26it [00:02, 11.17it/s, validationLossPerBatch=2.51]                                 
Epoch 146/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.76]                                                
Epoch 146/150: 26it [00:02, 10.73it/s, validationLossPerBatch=2.51]                                 
Epoch 147/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.77]                                                
Epoch 147/150: 26it [00:02, 11.14it/s, validationLossPerBatch=2.51]                                 
Epoch 148/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 148/150: 26it [00:02, 11.19it/s, validationLossPerBatch=2.52]                                 
Epoch 149/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 149/150: 26it [00:02, 11.10it/s, validationLossPerBatch=2.52]                                 
Epoch 150/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 150/150: 26it [00:02, 10.89it/s, validationLossPerBatch=2.52]  
Epoch 152/200: 184it [00:36,  5.07it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 152/200: 26it [00:02, 10.74it/s, validationLossPerBatch=2.52]                                 
Epoch 153/200: 184it [00:35,  5.14it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 153/200: 26it [00:02, 11.22it/s, validationLossPerBatch=2.51]                                 
Epoch 154/200: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 154/200: 26it [00:02, 10.93it/s, validationLossPerBatch=2.52]                                 
Epoch 155/200: 184it [00:35,  5.19it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 155/200: 26it [00:02, 11.34it/s, validationLossPerBatch=2.52]                                 
Epoch 156/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.7]                                                 
Epoch 156/200: 26it [00:02, 11.08it/s, validationLossPerBatch=2.51]                                 
Epoch 157/200: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 157/200: 26it [00:02, 10.49it/s, validationLossPerBatch=2.51]                                 
Epoch 158/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 158/200: 26it [00:02, 10.55it/s, validationLossPerBatch=2.5]                                  
[2025-05-22T23:24:33.771488|base.py:165] Caching best weights at epoch 158...
Epoch 159/200: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 159/200: 26it [00:02, 11.15it/s, validationLossPerBatch=2.51]                                 
Epoch 160/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.66]                                                
Epoch 160/200: 26it [00:02, 11.54it/s, validationLossPerBatch=2.5]                                  
[2025-05-22T23:25:48.449143|base.py:165] Caching best weights at epoch 160...
Epoch 161/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 161/200: 26it [00:02, 11.11it/s, validationLossPerBatch=2.51]                                 
Epoch 162/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.66]                                                
Epoch 162/200: 26it [00:02, 11.60it/s, validationLossPerBatch=2.5]                                  
[2025-05-22T23:27:02.817354|base.py:165] Caching best weights at epoch 162...
Epoch 163/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 163/200: 26it [00:02, 11.21it/s, validationLossPerBatch=2.52]                                 
Epoch 164/200: 184it [00:35,  5.12it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 164/200: 26it [00:02, 11.14it/s, validationLossPerBatch=2.51]                                 
Epoch 165/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 165/200: 26it [00:02, 10.92it/s, validationLossPerBatch=2.52]                                 
Epoch 166/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 166/200: 26it [00:02, 10.54it/s, validationLossPerBatch=2.51]                                 
Epoch 167/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 167/200: 26it [00:02, 10.68it/s, validationLossPerBatch=2.51]                                 
Epoch 168/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 168/200: 26it [00:02, 11.04it/s, validationLossPerBatch=2.51]                                 
Epoch 169/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 169/200: 26it [00:04,  6.39it/s, validationLossPerBatch=2.51]                                 
Epoch 170/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 170/200: 26it [00:02, 11.20it/s, validationLossPerBatch=2.5]                                  
Epoch 171/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.67]                                                
Epoch 171/200: 26it [00:02, 11.41it/s, validationLossPerBatch=2.49]                                 
[2025-05-22T23:32:41.007480|base.py:165] Caching best weights at epoch 171...
Epoch 172/200: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.69]                                                
Epoch 172/200: 26it [00:02, 10.93it/s, validationLossPerBatch=2.5]                                  
Epoch 173/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 173/200: 26it [00:02, 10.92it/s, validationLossPerBatch=2.51]                                 
Epoch 174/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.66]                                                
Epoch 174/200: 26it [00:02, 11.39it/s, validationLossPerBatch=2.5]                                  
Epoch 175/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.67]                                                
Epoch 175/200: 26it [00:02, 10.71it/s, validationLossPerBatch=2.5]                                  
Epoch 176/200: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 176/200: 26it [00:02, 11.00it/s, validationLossPerBatch=2.51]                                 
Epoch 177/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.67]                                                
Epoch 177/200: 26it [00:02, 10.76it/s, validationLossPerBatch=2.52]                                 
Epoch 178/200: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.67]                                                
Epoch 178/200: 26it [00:02, 11.43it/s, validationLossPerBatch=2.5]                                  
Epoch 179/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.67]                                                
Epoch 179/200: 26it [00:02, 10.93it/s, validationLossPerBatch=2.51]                                 
Epoch 180/200: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 180/200: 26it [00:02, 10.64it/s, validationLossPerBatch=2.51]                                 
Epoch 181/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.68]                                                
Epoch 181/200: 26it [00:02, 11.17it/s, validationLossPerBatch=2.5]                                  
Epoch 182/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.7]                                                 
Epoch 182/200: 26it [00:02, 11.01it/s, validationLossPerBatch=2.52]                                 
Epoch 183/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 183/200: 26it [00:02, 11.19it/s, validationLossPerBatch=2.51]                                 
Epoch 184/200: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 184/200: 26it [00:02, 11.36it/s, validationLossPerBatch=2.5]                                  
Epoch 185/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.7]                                                 
Epoch 185/200: 26it [00:02,  9.41it/s, validationLossPerBatch=2.53]                                 
Epoch 186/200: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 186/200: 26it [00:02, 11.20it/s, validationLossPerBatch=2.52]                                 
Epoch 187/200: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 187/200: 26it [00:02, 11.50it/s, validationLossPerBatch=2.5]                                  
Epoch 188/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 188/200: 26it [00:02, 11.30it/s, validationLossPerBatch=2.52]                                 
Epoch 189/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 189/200: 26it [00:02, 10.94it/s, validationLossPerBatch=2.5]                                  
Epoch 190/200: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 190/200: 26it [00:02, 11.46it/s, validationLossPerBatch=2.51]                                 
Epoch 191/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 191/200: 26it [00:02, 10.83it/s, validationLossPerBatch=2.52]                                 
Epoch 192/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 192/200: 26it [00:02, 11.55it/s, validationLossPerBatch=2.51]                                 
Epoch 193/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.71]                                                
Epoch 193/200: 26it [00:02, 10.78it/s, validationLossPerBatch=2.52]                                 
Epoch 194/200: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 194/200: 26it [00:02, 11.03it/s, validationLossPerBatch=2.5]                                  
Epoch 195/200: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.72]                                                
Epoch 195/200: 26it [00:02, 11.30it/s, validationLossPerBatch=2.5]                                  
Epoch 196/200: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 196/200: 26it [00:02, 11.27it/s, validationLossPerBatch=2.51]                                 
Epoch 197/200: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 197/200: 26it [00:02, 10.99it/s, validationLossPerBatch=2.5]                                  
Epoch 198/200: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.75]                                                
Epoch 198/200: 26it [00:02, 11.14it/s, validationLossPerBatch=2.51]                                 
Epoch 199/200: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.74]                                                
Epoch 199/200: 26it [00:02, 11.36it/s, validationLossPerBatch=2.51]                                 
Epoch 200/200: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.73]                                                
Epoch 200/200: 26it [00:02, 11.30it/s, validationLossPerBatch=2.51]                         
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