import config as config

from trainer  import Trainer,get_model_params


if __name__ == '__main__':

    for DL_FRAMEWORK in ['pytorch', 'tensorflow']:
        for MODELNAME in ['YetAnotherTransformer','TransformerPredictor',
                          'LSTMPredictor','TransformerEnsemble']:
            config.MODELNAME = MODELNAME
            config.DL_FRAMEWORK = DL_FRAMEWORK


            trainer = Trainer(config=config,
                              modelname=config.MODELNAME,
                              enableAugmentationDropout=False,
                              augmentation_threshold=0.35)
            # trainer.add_callback(dropout_callback)
            # trainer.add_callback(augmentation_increase_callback)
            trainer.train()
            trainer.test()


    #
    #
    # for BATCH_SIZE in [64,128,256]:
    #     for augmentation_threshold in [0.1,0.15,0.2,0.25,0.3]:
    #         config.BATCH_SIZE = BATCH_SIZE
    #         model_params = get_model_params(config.MODELNAME)
    #         for hidden_dim in [50,75,100,125,150]:
    #             for dropout in [0.2,0.25,.3,.35,.4,.45,.5,.6]:
    #                 model_params["params"]['hidden_dim'] = hidden_dim
    #                 model_params["params"]['dropout'] = dropout
    #                 # Get Data
    #                 trainer = Trainer(config=config,
    #                                   modelname=config.MODELNAME,
    #                                   enableAugmentationDropout=False,
    #                                   augmentation_threshold=augmentation_threshold,
    #                                   model_config=model_params)
    #                 # trainer.add_callback(dropout_callback)
    #                 # trainer.add_callback(augmentation_increase_callback)
    #                 trainer.train()
    #                 trainer.test()
    #
    #


