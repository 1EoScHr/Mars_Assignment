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

    # debug引入更改
    #mcfg.paintImages = True
    
    # 第一阶段
    """"""
    mcfg.baseLearningRate = 1e-2
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.9
    mcfg.optimizerWeightDecay = 5e-4
    
    # 第二阶段
    """
    mcfg.baseLearningRate = 3e-3
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.937
    mcfg.optimizerWeightDecay = 5e-4
    mcfg.startEpoch = 24
    """
     # 第三阶段
    """
    mcfg.baseLearningRate = 1e-3
    mcfg.minLearningRate = mcfg.baseLearningRate * 0.01
    mcfg.optimizerMomentum = 0.937
    mcfg.optimizerWeightDecay = 5e-4
    mcfg.startEpoch = 101
    """

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 120
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]

    if "teacher" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY

    if "distillation" in tags:
        mcfg.modelName = "distillation"
        mcfg.checkpointModelFile = "/auto/mars/ame/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.teacherModelFile = "/auto/mars/ame/c1.nano.teacher/__cache__/best_weights.pth"
        mcfg.distilLossWeights = (1.0, 0.05, 0.001)
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

    return mcfg
