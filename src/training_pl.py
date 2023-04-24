import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml
import importlib
from src.config import *
from src.data.data_utils import create_data_loaders
from src.data.dataset import ASL_DATASET

def main(args):

    # load model from YAML file
    if args.config_file:
        with open(args.config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # import model class
        module_name, class_name = yaml_config.pop('model').rsplit('.', 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # create model instance with remaining YAML arguments
        model = model_class(**yaml_config)


    # init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # train the model
    trainer.fit(model,
                data_module,
                )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--config-file', type=str, default=None, help='path to YAML config file')
    args = parser.parse_args()

    main(args)
