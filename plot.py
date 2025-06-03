import re
import matplotlib.pyplot as plt

success_log_text = """                                                                                                
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
log_text = """
Epoch 1/20: 184it [00:25,  7.33it/s, backboneFreezed=1, trainLossPerBatch=1.42e+3]                                                
Epoch 1/20: 26it [00:03,  7.57it/s, validationLossPerBatch=379]                                     
[2025-06-03T20:08:48.349607|base.py:181] Caching best weights at epoch 1...
Epoch 2/20: 184it [00:26,  6.88it/s, backboneFreezed=1, trainLossPerBatch=10.2]                                                   
Epoch 2/20: 26it [00:05,  4.75it/s, validationLossPerBatch=12.9]                                    
[2025-06-03T20:09:20.627644|base.py:181] Caching best weights at epoch 2...
Epoch 3/20: 184it [00:25,  7.22it/s, backboneFreezed=1, trainLossPerBatch=8.95]                                                   
Epoch 3/20: 26it [00:04,  6.42it/s, validationLossPerBatch=8.57]                                    
[2025-06-03T20:09:50.244454|base.py:181] Caching best weights at epoch 3...
Epoch 4/20: 184it [00:23,  7.78it/s, backboneFreezed=1, trainLossPerBatch=8.44]                                                   
Epoch 4/20: 26it [00:02, 10.16it/s, validationLossPerBatch=7.33]                                    
[2025-06-03T20:10:16.531270|base.py:181] Caching best weights at epoch 4...
Epoch 5/20: 184it [00:23,  7.92it/s, backboneFreezed=1, trainLossPerBatch=8.04]                                                   
Epoch 5/20: 26it [00:02, 10.88it/s, validationLossPerBatch=6.97]                                    
[2025-06-03T20:10:42.228831|base.py:181] Caching best weights at epoch 5...
Epoch 6/20: 184it [00:37,  4.93it/s, backboneFreezed=0, trainLossPerBatch=7.21]                                                   
Epoch 6/20: 26it [00:02,  9.92it/s, validationLossPerBatch=5.91]                                    
[2025-06-03T20:11:22.242705|base.py:181] Caching best weights at epoch 6...
Epoch 7/20: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=6.46]                                                   
Epoch 7/20: 26it [00:02, 10.36it/s, validationLossPerBatch=5.36]                                    
[2025-06-03T20:11:59.508433|base.py:181] Caching best weights at epoch 7...
Epoch 8/20: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=6.08]                                                   
Epoch 8/20: 26it [00:02, 11.37it/s, validationLossPerBatch=5.09]                                    
[2025-06-03T20:12:36.771174|base.py:181] Caching best weights at epoch 8...
Epoch 9/20: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=5.77]                                                   
Epoch 9/20: 26it [00:02, 11.27it/s, validationLossPerBatch=4.84]                                    
[2025-06-03T20:13:13.913505|base.py:181] Caching best weights at epoch 9...
Epoch 10/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=5.61]                                                  
Epoch 10/20: 26it [00:02, 10.84it/s, validationLossPerBatch=4.67]                                   
[2025-06-03T20:13:51.209769|base.py:181] Caching best weights at epoch 10...
Epoch 11/20: 184it [00:35,  5.26it/s, backboneFreezed=0, trainLossPerBatch=5.38]                                                  
Epoch 11/20: 26it [00:02, 10.78it/s, validationLossPerBatch=4.54]                                   
[2025-06-03T20:14:28.690994|base.py:181] Caching best weights at epoch 11...
Epoch 12/20: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=5.25]                                                  
Epoch 12/20: 26it [00:02, 11.12it/s, validationLossPerBatch=4.42]                                   
[2025-06-03T20:15:06.047838|base.py:181] Caching best weights at epoch 12...
Epoch 13/20: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=5.11]                                                  
Epoch 13/20: 26it [00:02, 11.03it/s, validationLossPerBatch=4.29]                                   
[2025-06-03T20:15:43.399529|base.py:181] Caching best weights at epoch 13...
Epoch 14/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=5]                                                     
Epoch 14/20: 26it [00:02, 11.12it/s, validationLossPerBatch=4.24]                                   
[2025-06-03T20:16:20.624651|base.py:181] Caching best weights at epoch 14...
Epoch 15/20: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=4.86]                                                  
Epoch 15/20: 26it [00:02, 11.21it/s, validationLossPerBatch=4.14]                                   
[2025-06-03T20:16:57.893129|base.py:181] Caching best weights at epoch 15...
Epoch 16/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.76]                                                  
Epoch 16/20: 26it [00:02, 10.87it/s, validationLossPerBatch=4.08]                                   
[2025-06-03T20:17:35.189244|base.py:181] Caching best weights at epoch 16...
Epoch 17/20: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.71]                                                  
Epoch 17/20: 26it [00:02, 10.86it/s, validationLossPerBatch=4.01]                                   
[2025-06-03T20:18:12.485117|base.py:181] Caching best weights at epoch 17...
Epoch 18/20: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=4.63]                                                  
Epoch 18/20: 26it [00:02, 10.29it/s, validationLossPerBatch=4]                                      
[2025-06-03T20:18:50.052055|base.py:181] Caching best weights at epoch 18...
Epoch 19/20: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=4.6]                                                   
Epoch 19/20: 26it [00:02, 10.88it/s, validationLossPerBatch=3.99]                                   
[2025-06-03T20:19:27.162239|base.py:181] Caching best weights at epoch 19...
Epoch 20/20: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=4.55]                                                  
Epoch 20/20: 26it [00:02, 10.85it/s, validationLossPerBatch=3.95]                                   
[2025-06-03T20:20:04.569193|base.py:181] Caching best weights at epoch 20...
Epoch 21/100: 184it [00:37,  4.92it/s, backboneFreezed=0, trainLossPerBatch=4.88]                                                 
Epoch 21/100: 26it [00:02,  9.62it/s, validationLossPerBatch=3.98]                                  
Epoch 22/100: 184it [00:36,  5.03it/s, backboneFreezed=0, trainLossPerBatch=4.86]                                                 
Epoch 22/100: 26it [00:02, 11.79it/s, validationLossPerBatch=3.91]                                  
[2025-06-03T20:23:03.731669|base.py:181] Caching best weights at epoch 22...
Epoch 23/100: 184it [00:35,  5.26it/s, backboneFreezed=0, trainLossPerBatch=4.74]                                                 
Epoch 23/100: 26it [00:02, 11.68it/s, validationLossPerBatch=3.85]                                  
[2025-06-03T20:23:41.034377|base.py:181] Caching best weights at epoch 23...
Epoch 24/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=4.64]                                                 
Epoch 24/100: 26it [00:02, 12.03it/s, validationLossPerBatch=3.78]                                  
[2025-06-03T20:24:17.920480|base.py:181] Caching best weights at epoch 24...
Epoch 25/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.61]                                                 
Epoch 25/100: 26it [00:02, 10.71it/s, validationLossPerBatch=3.69]                                  
[2025-06-03T20:24:55.136615|base.py:181] Caching best weights at epoch 25...
Epoch 26/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=4.54]                                                 
Epoch 26/100: 26it [00:02, 11.10it/s, validationLossPerBatch=3.64]                                  
[2025-06-03T20:25:32.332111|base.py:181] Caching best weights at epoch 26...
Epoch 27/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=4.5]                                                  
Epoch 27/100: 26it [00:02, 11.41it/s, validationLossPerBatch=3.59]                                  
[2025-06-03T20:26:09.593874|base.py:181] Caching best weights at epoch 27...
Epoch 28/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.46]                                                 
Epoch 28/100: 26it [00:02, 11.39it/s, validationLossPerBatch=3.57]                                  
[2025-06-03T20:26:46.757619|base.py:181] Caching best weights at epoch 28...
Epoch 29/100: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=4.35]                                                 
Epoch 29/100: 26it [00:02, 11.31it/s, validationLossPerBatch=3.49]                                  
[2025-06-03T20:27:24.220420|base.py:181] Caching best weights at epoch 29...
Epoch 30/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.32]                                                 
Epoch 30/100: 26it [00:02, 11.59it/s, validationLossPerBatch=3.44]                                  
[2025-06-03T20:28:01.235730|base.py:181] Caching best weights at epoch 30...
Epoch 31/100: 184it [00:34,  5.32it/s, backboneFreezed=0, trainLossPerBatch=4.26]                                                 
Epoch 31/100: 26it [00:02, 11.09it/s, validationLossPerBatch=3.39]                                  
[2025-06-03T20:28:38.213068|base.py:181] Caching best weights at epoch 31...
Epoch 32/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=4.22]                                                 
Epoch 32/100: 26it [00:02, 11.16it/s, validationLossPerBatch=3.38]                                  
[2025-06-03T20:29:15.304574|base.py:181] Caching best weights at epoch 32...
Epoch 33/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=4.18]                                                 
Epoch 33/100: 26it [00:02, 11.70it/s, validationLossPerBatch=3.33]                                  
[2025-06-03T20:29:52.254253|base.py:181] Caching best weights at epoch 33...
Epoch 34/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=4.17]                                                 
Epoch 34/100: 26it [00:02, 11.33it/s, validationLossPerBatch=3.34]                                  
Epoch 35/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.1]                                                  
Epoch 35/100: 26it [00:02, 11.43it/s, validationLossPerBatch=3.28]                                  
[2025-06-03T20:31:06.535981|base.py:181] Caching best weights at epoch 35...
Epoch 36/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=4.06]                                                 
Epoch 36/100: 26it [00:02, 11.34it/s, validationLossPerBatch=3.23]                                  
[2025-06-03T20:31:43.729903|base.py:181] Caching best weights at epoch 36...
Epoch 37/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=4.04]                                                 
Epoch 37/100: 26it [00:02, 11.14it/s, validationLossPerBatch=3.17]                                  
[2025-06-03T20:32:20.906209|base.py:181] Caching best weights at epoch 37...
Epoch 38/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=4]                                                    
Epoch 38/100: 26it [00:02, 11.67it/s, validationLossPerBatch=3.16]                                  
[2025-06-03T20:32:57.976257|base.py:181] Caching best weights at epoch 38...
Epoch 39/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.95]                                                 
Epoch 39/100: 26it [00:02, 11.63it/s, validationLossPerBatch=3.12]                                  
[2025-06-03T20:33:34.959453|base.py:181] Caching best weights at epoch 39...
Epoch 40/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.91]                                                 
Epoch 40/100: 26it [00:02, 11.14it/s, validationLossPerBatch=3.09]                                  
[2025-06-03T20:34:12.122093|base.py:181] Caching best weights at epoch 40...
Epoch 41/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.9]                                                  
Epoch 41/100: 26it [00:02, 11.48it/s, validationLossPerBatch=3.07]                                  
[2025-06-03T20:34:49.190422|base.py:181] Caching best weights at epoch 41...
Epoch 42/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.87]                                                 
Epoch 42/100: 26it [00:02, 11.82it/s, validationLossPerBatch=3.05]                                  
[2025-06-03T20:35:26.317737|base.py:181] Caching best weights at epoch 42...
Epoch 43/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.8]                                                  
Epoch 43/100: 26it [00:02, 11.88it/s, validationLossPerBatch=3.01]                                  
[2025-06-03T20:36:03.297226|base.py:181] Caching best weights at epoch 43...
Epoch 44/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.75]                                                 
Epoch 44/100: 26it [00:02, 11.53it/s, validationLossPerBatch=2.98]                                  
[2025-06-03T20:36:40.436574|base.py:181] Caching best weights at epoch 44...
Epoch 45/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.74]                                                 
Epoch 45/100: 26it [00:02, 11.59it/s, validationLossPerBatch=2.97]                                  
[2025-06-03T20:37:17.612099|base.py:181] Caching best weights at epoch 45...
Epoch 46/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.71]                                                 
Epoch 46/100: 26it [00:02, 11.59it/s, validationLossPerBatch=2.95]                                  
[2025-06-03T20:37:54.734419|base.py:181] Caching best weights at epoch 46...
Epoch 47/100: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=3.7]                                                  
Epoch 47/100: 26it [00:02, 11.31it/s, validationLossPerBatch=2.92]                                  
[2025-06-03T20:38:32.071957|base.py:181] Caching best weights at epoch 47...
Epoch 48/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.67]                                                 
Epoch 48/100: 26it [00:02, 11.71it/s, validationLossPerBatch=2.94]                                  
Epoch 49/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.62]                                                 
Epoch 49/100: 26it [00:02, 11.87it/s, validationLossPerBatch=2.9]                                   
[2025-06-03T20:39:46.114141|base.py:181] Caching best weights at epoch 49...
Epoch 50/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.59]                                                 
Epoch 50/100: 26it [00:02, 11.51it/s, validationLossPerBatch=2.86]                                  
[2025-06-03T20:40:23.133838|base.py:181] Caching best weights at epoch 50...
Epoch 51/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.58]                                                 
Epoch 51/100: 26it [00:02, 11.27it/s, validationLossPerBatch=2.89]                                  
Epoch 52/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.58]                                                 
Epoch 52/100: 26it [00:02, 11.75it/s, validationLossPerBatch=2.84]                                  
[2025-06-03T20:41:37.517293|base.py:181] Caching best weights at epoch 52...
Epoch 53/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.54]                                                 
Epoch 53/100: 26it [00:02, 11.60it/s, validationLossPerBatch=2.81]                                  
[2025-06-03T20:42:14.542044|base.py:181] Caching best weights at epoch 53...
Epoch 54/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.49]                                                 
Epoch 54/100: 26it [00:02, 10.87it/s, validationLossPerBatch=2.84]                                  
Epoch 55/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.48]                                                 
Epoch 55/100: 26it [00:02, 11.65it/s, validationLossPerBatch=2.79]                                  
[2025-06-03T20:43:28.578464|base.py:181] Caching best weights at epoch 55...
Epoch 56/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.48]                                                 
Epoch 56/100: 26it [00:02, 11.81it/s, validationLossPerBatch=2.76]                                  
[2025-06-03T20:44:05.713297|base.py:181] Caching best weights at epoch 56...
Epoch 57/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.43]                                                 
Epoch 57/100: 26it [00:02, 11.41it/s, validationLossPerBatch=2.78]                                  
Epoch 58/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.41]                                                 
Epoch 58/100: 26it [00:02, 11.62it/s, validationLossPerBatch=2.75]                                  
[2025-06-03T20:45:19.835610|base.py:181] Caching best weights at epoch 58...
Epoch 59/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.41]                                                 
Epoch 59/100: 26it [00:02, 11.78it/s, validationLossPerBatch=2.76]                                  
Epoch 60/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.36]                                                 
Epoch 60/100: 26it [00:02, 11.54it/s, validationLossPerBatch=2.75]                                  
[2025-06-03T20:46:33.936028|base.py:181] Caching best weights at epoch 60...
Epoch 61/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.36]                                                 
Epoch 61/100: 26it [00:02, 11.22it/s, validationLossPerBatch=2.71]                                  
[2025-06-03T20:47:11.229787|base.py:181] Caching best weights at epoch 61...
Epoch 62/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.32]                                                 
Epoch 62/100: 26it [00:02, 11.47it/s, validationLossPerBatch=2.72]                                  
Epoch 63/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.31]                                                 
Epoch 63/100: 26it [00:02, 11.56it/s, validationLossPerBatch=2.69]                                  
[2025-06-03T20:48:25.418554|base.py:181] Caching best weights at epoch 63...
Epoch 64/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.27]                                                 
Epoch 64/100: 26it [00:02, 12.07it/s, validationLossPerBatch=2.68]                                  
[2025-06-03T20:49:02.466188|base.py:181] Caching best weights at epoch 64...
Epoch 65/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.26]                                                 
Epoch 65/100: 26it [00:02, 10.98it/s, validationLossPerBatch=2.68]                                  
Epoch 66/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.26]                                                 
Epoch 66/100: 26it [00:02, 11.30it/s, validationLossPerBatch=2.68]                                  
Epoch 67/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.25]                                                 
Epoch 67/100: 26it [00:02, 11.92it/s, validationLossPerBatch=2.67]                                  
[2025-06-03T20:50:53.750413|base.py:181] Caching best weights at epoch 67...
Epoch 68/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.22]                                                 
Epoch 68/100: 26it [00:02, 11.60it/s, validationLossPerBatch=2.67]                                  
[2025-06-03T20:51:30.992199|base.py:181] Caching best weights at epoch 68...
Epoch 69/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.21]                                                 
Epoch 69/100: 26it [00:02, 11.46it/s, validationLossPerBatch=2.66]                                  
[2025-06-03T20:52:08.190498|base.py:181] Caching best weights at epoch 69...
Epoch 70/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.18]                                                 
Epoch 70/100: 26it [00:02, 10.98it/s, validationLossPerBatch=2.66]                                  
[2025-06-03T20:52:45.269112|base.py:181] Caching best weights at epoch 70...
Epoch 71/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.15]                                                 
Epoch 71/100: 26it [00:02, 12.13it/s, validationLossPerBatch=2.66]                                  
[2025-06-03T20:53:22.299828|base.py:181] Caching best weights at epoch 71...
Epoch 72/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.18]                                                 
Epoch 72/100: 26it [00:02, 11.42it/s, validationLossPerBatch=2.64]                                  
[2025-06-03T20:53:59.289281|base.py:181] Caching best weights at epoch 72...
Epoch 73/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.16]                                                 
Epoch 73/100: 26it [00:02, 11.12it/s, validationLossPerBatch=2.63]                                  
[2025-06-03T20:54:36.524559|base.py:181] Caching best weights at epoch 73...
Epoch 74/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.13]                                                 
Epoch 74/100: 26it [00:02, 11.58it/s, validationLossPerBatch=2.63]                                  
[2025-06-03T20:55:13.528581|base.py:181] Caching best weights at epoch 74...
Epoch 75/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.1]                                                  
Epoch 75/100: 26it [00:02, 11.85it/s, validationLossPerBatch=2.62]                                  
[2025-06-03T20:55:50.440204|base.py:181] Caching best weights at epoch 75...
Epoch 76/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.14]                                                 
Epoch 76/100: 26it [00:02, 11.14it/s, validationLossPerBatch=2.62]                                  
Epoch 77/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.1]                                                  
Epoch 77/100: 26it [00:02, 11.82it/s, validationLossPerBatch=2.6]                                   
[2025-06-03T20:57:04.774660|base.py:181] Caching best weights at epoch 77...
Epoch 78/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.06]                                                 
Epoch 78/100: 26it [00:02, 10.98it/s, validationLossPerBatch=2.6]                                   
Epoch 79/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.07]                                                 
Epoch 79/100: 26it [00:02, 11.93it/s, validationLossPerBatch=2.6]                                   
[2025-06-03T20:58:18.801672|base.py:181] Caching best weights at epoch 79...
Epoch 80/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.04]                                                 
Epoch 80/100: 26it [00:02, 11.81it/s, validationLossPerBatch=2.6]                                   
[2025-06-03T20:58:55.700669|base.py:181] Caching best weights at epoch 80...
Epoch 81/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.06]                                                 
Epoch 81/100: 26it [00:02, 11.64it/s, validationLossPerBatch=2.6]                                   
Epoch 82/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.04]                                                 
Epoch 82/100: 26it [00:02, 11.18it/s, validationLossPerBatch=2.62]                                  
Epoch 83/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3.03]                                                 
Epoch 83/100: 26it [00:02, 11.59it/s, validationLossPerBatch=2.59]                                  
[2025-06-03T21:00:46.835774|base.py:181] Caching best weights at epoch 83...
Epoch 84/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.04]                                                 
Epoch 84/100: 26it [00:02, 11.68it/s, validationLossPerBatch=2.58]                                  
[2025-06-03T21:01:23.878067|base.py:181] Caching best weights at epoch 84...
Epoch 85/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.03]                                                 
Epoch 85/100: 26it [00:02, 11.31it/s, validationLossPerBatch=2.58]                                  
[2025-06-03T21:02:01.139292|base.py:181] Caching best weights at epoch 85...
Epoch 86/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.01]                                                 
Epoch 86/100: 26it [00:02, 10.70it/s, validationLossPerBatch=2.59]                                  
Epoch 87/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3.04]                                                 
Epoch 87/100: 26it [00:02, 11.73it/s, validationLossPerBatch=2.59]                                  
Epoch 88/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3.01]                                                 
Epoch 88/100: 26it [00:02, 11.43it/s, validationLossPerBatch=2.58]                                  
[2025-06-03T21:03:52.452845|base.py:181] Caching best weights at epoch 88...
Epoch 89/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.98]                                                 
Epoch 89/100: 26it [00:02, 11.68it/s, validationLossPerBatch=2.59]                                  
Epoch 90/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.01]                                                 
Epoch 90/100: 26it [00:02, 11.77it/s, validationLossPerBatch=2.57]                                  
[2025-06-03T21:05:06.818512|base.py:181] Caching best weights at epoch 90...
Epoch 91/100: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=3.01]                                                 
Epoch 91/100: 26it [00:02, 11.40it/s, validationLossPerBatch=2.57]                                  
Epoch 92/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3]                                                    
Epoch 92/100: 26it [00:02, 11.23it/s, validationLossPerBatch=2.59]                                  
Epoch 93/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3]                                                    
Epoch 93/100: 26it [00:02, 10.81it/s, validationLossPerBatch=2.58]                                  
Epoch 94/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.99]                                                 
Epoch 94/100: 26it [00:02, 11.43it/s, validationLossPerBatch=2.58]                                  
Epoch 95/100: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3]                                                    
Epoch 95/100: 26it [00:02, 11.52it/s, validationLossPerBatch=2.58]                                  
Epoch 96/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.98]                                                 
Epoch 96/100: 26it [00:02, 11.61it/s, validationLossPerBatch=2.58]                                  
Epoch 97/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=2.97]                                                 
Epoch 97/100: 26it [00:02, 11.81it/s, validationLossPerBatch=2.58]                                  
Epoch 98/100: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3]                                                    
Epoch 98/100: 26it [00:02, 11.07it/s, validationLossPerBatch=2.57]                                  
[2025-06-03T21:10:03.374331|base.py:181] Caching best weights at epoch 98...
Epoch 99/100: 184it [00:34,  5.31it/s, backboneFreezed=0, trainLossPerBatch=3.02]                                                 
Epoch 99/100: 26it [00:02, 11.74it/s, validationLossPerBatch=2.57]                                  
Epoch 100/100: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=3]                                                   
Epoch 100/100: 26it [00:02, 11.32it/s, validationLossPerBatch=2.57]       
Epoch 101/150: 184it [00:35,  5.13it/s, backboneFreezed=0, trainLossPerBatch=3]                                                   
Epoch 101/150: 26it [00:02, 10.85it/s, validationLossPerBatch=2.58]                                 
Epoch 102/150: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=3.03]                                                
Epoch 102/150: 26it [00:02, 11.80it/s, validationLossPerBatch=2.58]                                 
Epoch 103/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=3]                                                   
Epoch 103/150: 26it [00:02, 11.95it/s, validationLossPerBatch=2.59]                                 
Epoch 104/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.98]                                                
Epoch 104/150: 26it [00:02, 10.77it/s, validationLossPerBatch=2.57]                                 
Epoch 105/150: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=3.01]                                                
Epoch 105/150: 26it [00:02, 11.11it/s, validationLossPerBatch=2.57]                                 
Epoch 106/150: 184it [00:35,  5.21it/s, backboneFreezed=0, trainLossPerBatch=2.98]                                                
Epoch 106/150: 26it [00:02, 10.74it/s, validationLossPerBatch=2.57]                                 
Epoch 107/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.99]                                                
Epoch 107/150: 26it [00:02, 11.24it/s, validationLossPerBatch=2.56]                                 
[2025-06-03T21:18:23.502591|base.py:181] Caching best weights at epoch 107...
Epoch 108/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=3]                                                   
Epoch 108/150: 26it [00:02, 11.98it/s, validationLossPerBatch=2.58]                                 
Epoch 109/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.94]                                                
Epoch 109/150: 26it [00:02, 10.92it/s, validationLossPerBatch=2.55]                                 
[2025-06-03T21:19:37.854546|base.py:181] Caching best weights at epoch 109...
Epoch 110/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.98]                                                
Epoch 110/150: 26it [00:02, 11.57it/s, validationLossPerBatch=2.57]                                 
Epoch 111/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.94]                                                
Epoch 111/150: 26it [00:02, 11.03it/s, validationLossPerBatch=2.56]                                 
Epoch 112/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.97]                                                
Epoch 112/150: 26it [00:02, 11.82it/s, validationLossPerBatch=2.58]                                 
Epoch 113/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.97]                                                
Epoch 113/150: 26it [00:02, 11.48it/s, validationLossPerBatch=2.56]                                 
Epoch 114/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.97]                                                
Epoch 114/150: 26it [00:02, 11.41it/s, validationLossPerBatch=2.58]                                 
Epoch 115/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.96]                                                
Epoch 115/150: 26it [00:02, 11.19it/s, validationLossPerBatch=2.57]                                 
Epoch 116/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.96]                                                
Epoch 116/150: 26it [00:02, 11.19it/s, validationLossPerBatch=2.56]                                 
Epoch 117/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                
Epoch 117/150: 26it [00:02, 11.02it/s, validationLossPerBatch=2.55]                                 
Epoch 118/150: 184it [00:35,  5.24it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                
Epoch 118/150: 26it [00:02, 11.54it/s, validationLossPerBatch=2.56]                                 
Epoch 119/150: 184it [00:35,  5.21it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                
Epoch 119/150: 26it [00:02, 11.61it/s, validationLossPerBatch=2.55]                                 
Epoch 120/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.93]                                                
Epoch 120/150: 26it [00:02, 12.06it/s, validationLossPerBatch=2.54]                                 
[2025-06-03T21:26:27.158123|base.py:181] Caching best weights at epoch 120...
Epoch 121/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.95]                                                
Epoch 121/150: 26it [00:02, 11.47it/s, validationLossPerBatch=2.55]                                 
Epoch 122/150: 184it [00:35,  5.22it/s, backboneFreezed=0, trainLossPerBatch=2.93]                                                
Epoch 122/150: 26it [00:02, 11.52it/s, validationLossPerBatch=2.55]                                 
Epoch 123/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.92]                                                
Epoch 123/150: 26it [00:02, 10.89it/s, validationLossPerBatch=2.55]                                 
Epoch 124/150: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 124/150: 26it [00:02, 12.03it/s, validationLossPerBatch=2.54]                                 
Epoch 125/150: 184it [00:34,  5.30it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                
Epoch 125/150: 26it [00:02, 11.64it/s, validationLossPerBatch=2.56]                                 
Epoch 126/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 126/150: 26it [00:02, 11.34it/s, validationLossPerBatch=2.56]                                 
Epoch 127/150: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 127/150: 26it [00:02, 11.78it/s, validationLossPerBatch=2.54]                                 
Epoch 128/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.92]                                                
Epoch 128/150: 26it [00:02, 11.05it/s, validationLossPerBatch=2.56]                                 
Epoch 129/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                
Epoch 129/150: 26it [00:02, 11.44it/s, validationLossPerBatch=2.55]                                 
Epoch 130/150: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.89]                                                
Epoch 130/150: 26it [00:02, 11.59it/s, validationLossPerBatch=2.54]                                 
[2025-06-03T21:32:39.799330|base.py:181] Caching best weights at epoch 130...
Epoch 131/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                
Epoch 131/150: 26it [00:02, 11.22it/s, validationLossPerBatch=2.56]                                 
Epoch 132/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                
Epoch 132/150: 26it [00:02, 10.99it/s, validationLossPerBatch=2.55]                                 
Epoch 133/150: 184it [00:35,  5.25it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 133/150: 26it [00:02, 11.60it/s, validationLossPerBatch=2.54]                                 
Epoch 134/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 134/150: 26it [00:02, 11.34it/s, validationLossPerBatch=2.57]                                 
Epoch 135/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 135/150: 26it [00:02, 11.46it/s, validationLossPerBatch=2.55]                                 
Epoch 136/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.91]                                                
Epoch 136/150: 26it [00:02, 10.91it/s, validationLossPerBatch=2.55]                                 
Epoch 137/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 137/150: 26it [00:02, 10.87it/s, validationLossPerBatch=2.57]                                 
Epoch 138/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 138/150: 26it [00:02, 11.54it/s, validationLossPerBatch=2.54]                                 
Epoch 139/150: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 139/150: 26it [00:02, 11.81it/s, validationLossPerBatch=2.54]                                 
Epoch 140/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 140/150: 26it [00:02, 11.27it/s, validationLossPerBatch=2.55]                                 
Epoch 141/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 141/150: 26it [00:02, 11.74it/s, validationLossPerBatch=2.54]                                 
Epoch 142/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 142/150: 26it [00:02, 11.53it/s, validationLossPerBatch=2.56]                                 
Epoch 143/150: 184it [00:34,  5.28it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 143/150: 26it [00:02, 11.65it/s, validationLossPerBatch=2.54]                                 
Epoch 144/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.86]                                                
Epoch 144/150: 26it [00:02, 11.27it/s, validationLossPerBatch=2.54]                                 
Epoch 145/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 145/150: 26it [00:02, 10.58it/s, validationLossPerBatch=2.55]                                 
Epoch 146/150: 184it [00:34,  5.26it/s, backboneFreezed=0, trainLossPerBatch=2.89]                                                
Epoch 146/150: 26it [00:02, 11.24it/s, validationLossPerBatch=2.54]                                 
Epoch 147/150: 184it [00:35,  5.23it/s, backboneFreezed=0, trainLossPerBatch=2.9]                                                 
Epoch 147/150: 26it [00:02, 11.59it/s, validationLossPerBatch=2.55]                                 
Epoch 148/150: 184it [00:34,  5.27it/s, backboneFreezed=0, trainLossPerBatch=2.88]                                                
Epoch 148/150: 26it [00:02, 11.38it/s, validationLossPerBatch=2.55]                                 
Epoch 149/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.87]                                                
Epoch 149/150: 26it [00:02, 11.55it/s, validationLossPerBatch=2.56]                                 
Epoch 150/150: 184it [00:34,  5.29it/s, backboneFreezed=0, trainLossPerBatch=2.86]                                                
Epoch 150/150: 26it [00:02, 11.46it/s, validationLossPerBatch=2.55]                
"""

# 画单个图
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
"""
def extract_losses(log_text):
    train_pattern = re.compile(r"Epoch (\d+)/\d+: \d+it \[.*?trainLossPerBatch=([\d\.]+)\]")
    val_pattern = re.compile(r"Epoch (\d+)/\d+: \d+it \[.*?validationLossPerBatch=([\d\.]+)\]")

    train_losses = {}
    val_losses = {}

    for line in log_text.strip().splitlines():
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

    return train_losses, val_losses

# 提取两个日志的损失
train_A, val_A = extract_losses(success_log_text)
train_B, val_B = extract_losses(log_text)

# 各自的 epoch 列表（只画共有的 epoch）
epochs = sorted(set(train_A) & set(val_A) & set(train_B) & set(val_B))

# 组装画图数据
train_loss_A = [train_A[e] for e in epochs]
val_loss_A = [val_A[e] for e in epochs]
train_loss_B = [train_B[e] for e in epochs]
val_loss_B = [val_B[e] for e in epochs]

# 画图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_A, 'o-', label="Train Loss Base")
plt.plot(epochs, val_loss_A, 'o--', label="Validation Loss Base")
plt.plot(epochs, train_loss_B, 's-', label="Train Loss EMA")
plt.plot(epochs, val_loss_B, 's--', label="Validation Loss EMA")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison.png")
print("Loss comparison curve saved as loss_comparison.png")
