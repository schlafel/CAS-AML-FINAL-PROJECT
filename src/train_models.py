import config as config

from trainer  import Trainer,get_model_params


if __name__ == '__main__':
    for BATCH_SIZE in [64,128,256]:
        for augmentation_threshold in [0.1,0.15,0.2,0.25,0.3]:
            config.BATCH_SIZE = BATCH_SIZE
            model_params = get_model_params(config.MODELNAME)

            for hidden_dim in [50,75,100,125,150]:
                model_params["params"]['hidden_dim'] = hidden_dim
                # Get Data
                trainer = Trainer(config=config,
                                  modelname=config.MODELNAME,
                                  enableAugmentationDropout=False,
                                  augmentation_threshold=augmentation_threshold,
                                  model_config=model_params)
                # trainer.add_callback(dropout_callback)
                # trainer.add_callback(augmentation_increase_callback)
                trainer.train()
                trainer.test()




