import config as config

from trainer  import Trainer


if __name__ == '__main__':
    for BATCH_SIZE in [64,128,256]:
        for augmentation_threshold in [0.1,0.15,0.2,0.25,0.3]:
            config.BATCH_SIZE = BATCH_SIZE
            # Get Data
            trainer = Trainer(
                              modelname=config.MODELNAME,
                              enableAugmentationDropout=False,
                              augmentation_threshold=augmentation_threshold)
            # trainer.add_callback(dropout_callback)
            # trainer.add_callback(augmentation_increase_callback)
            trainer.train()
            trainer.test()




