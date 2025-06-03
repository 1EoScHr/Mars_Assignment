import os
from config import mconfig


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    #projectRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    #mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    mcfg.phase = "nano" # DO NOT MODIFY
    mcfg.trainSplitName = "train" # DO NOT MODIFY
    mcfg.validationSplitName = "validation" # DO NOT MODIFY
    mcfg.testSplitName = "test" # DO NOT MODIFY

    # data setup
    mcfg.imageDir = "../Mars_Assignment_Running/mar20/images"
    mcfg.annotationDir = "../Mars_Assignment_Running/mar20/annotations"
    mcfg.classList = ["A{}".format(x) for x in range(1, 21)] # DO NOT MODIFY
    mcfg.subsetMap = { # DO NOT MODIFY
        "train": "../Mars_Assignment_Running/mar20/splits/v5/train.txt",
        "validation": "../Mars_Assignment_Running/mar20/splits/v5/validation.txt",
        "test": "../Mars_Assignment_Running/mar20/splits/v5/test.txt",
        "small": "../Mars_Assignment_Running/mar20/splits/v5/small.txt",
    }

    # 以下是在debug过程中引入的更改
    #mcfg.paintImages = True # 绘制最后识别效果，用于直观评估模型性能
    #mcfg.backbone_type = "swintransformer" # 使用backbone还是swintransformer
    mcfg.schedulerType = "plateau" # 使用动态调整学习率策略
    
    # 第一阶段
    """"""
    mcfg.baseLearningRate = 1e-3
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.9
    mcfg.optimizerWeightDecay = 5e-4
    
    # 第二阶段
    """
    mcfg.baseLearningRate = 1e-3 # 3e-3
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.937
    mcfg.optimizerWeightDecay = 5e-4
    """
    # 第三阶段
    """
    mcfg.baseLearningRate = 1e-4
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.937
    mcfg.optimizerWeightDecay = 5e-4
    mcfg.startEpoch = 151
    """

    if "ema" in tags:
        mcfg.useEMA = True

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]

    if "teacher" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 150
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY

    if "distillation" in tags:
        mcfg.modelName = "distillation"
        mcfg.checkpointModelFile = "../Mars_Assignment_Running/liuzt/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.teacherModelFile = "../Mars_Assignment_Running/liuzt/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.distilLossWeights = (1.0, 0.1, 0.05)#(1.0, 0.05, 0.001)
        mcfg.maxEpoch = 150
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

        mcfg.temperature = 2.0 # 自定义的温度

    return mcfg
