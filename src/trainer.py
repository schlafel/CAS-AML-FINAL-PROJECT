import sys,os
sys.path.insert(0, '../src')

from config import *
import numpy as np
from tqdm import tqdm

import datetime

from torch.utils.tensorboard import SummaryWriter
import time


class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, log_dir = f'./../checkpoints/{DL_FRAMEWORK}/{MODELNAME}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader

        self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs=EPOCHS):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}", flush=True)
            time.sleep(0.5)  # time to flush std out

            train_losses = []
            train_accuracies = []

            self.model.train_mode()

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                 desc=f"Training progress")

            total_loss = 0
            total_acc = 0

            for i, batch in pbar:

                loss, acc = self.model.training_step(batch)
                loss = loss.numpy()
                acc = acc.numpy()

                self.model.optimize()

                total_loss += loss
                total_acc += acc

                pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

                train_losses.append(loss)
                train_accuracies.append(acc)

            self.model.step_scheduler()

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)

            print(f"EPOCH {epoch+1:>3}: Train accuracy: {avg_train_acc:>3.2f}, Train Loss: {avg_train_loss:>9.8f}",
                  flush=True)



            val_loss, val_acc = self.evaluate()
            print(f"EPOCH {epoch+1:>3}: Validation accuracy: {val_acc:>3.2f}, Validation Loss: {val_loss:>9.8f}",
                  flush=True)

            print(flush=True)


            self.writer.add_scalars('loss',
                                    {'train':avg_train_loss,
                                     'val':val_loss}, global_step=epoch+1,)
            self.writer.add_scalars('accuracy',
                                    {'train':avg_train_acc,
                                     'val':val_acc}, global_step=epoch+1,)

            # self.writer.add_scalar('loss/val', val_loss, global_step=epoch+1,)
            # self.writer.add_scalar('accuracy/val', val_acc, global_step=epoch+1,)

    def evaluate(self):
        self.model.eval_mode()

        valid_losses = []
        valid_accuracies = []

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc=f"Validation progress")

        total_loss = 0
        total_acc = 0

        for i, batch in pbar:
            loss, acc = self.model.validation_step(batch)
            loss = loss.numpy()
            acc = acc.numpy()

            valid_losses.append(loss)
            valid_accuracies.append(acc)

            total_loss += loss
            total_acc += acc

            pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)

        return avg_valid_loss, avg_valid_acc

    def test(self,load_best = True):
        """
        Method to thest the model.
        :param load_best: Load best model before testing (based on val-accuracy)
        :type load_best: bool
        :return:
        """
        self.model.eval_mode()

        test_losses = []
        test_accuracies = []
        all_preds = []
        all_labels = []
        for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader),
                             desc=f"Testing progress"):
            loss, acc, preds = self.model.test_step(batch)
            loss = loss.numpy()
            acc = acc.numpy()
            preds = preds.numpy()


            test_losses.append(loss)
            test_accuracies.append(acc)
            all_preds.append(preds)
            all_labels.append(batch[1])

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)

        print(f"Test Accuracy: {avg_test_acc:>3.2f}, Test Loss: {avg_test_loss:>9.8f}")

        self.writer.add_scalar('test_loss', avg_test_loss, global_step=1, )
        self.writer.add_scalar('test_accuracy', avg_test_acc, global_step=1, )

        return all_preds, all_labels

import importlib
from data.dataset import ASL_DATASET
from data.data_utils import create_data_loaders
from dl_utils import get_model_params



if __name__ == '__main__':

    module_name = f"models.{DL_FRAMEWORK}.models"
    class_name = MODELNAME
    params = get_model_params(MODELNAME)

    print(f"Using model: {module_name}.{class_name}")

    module = importlib.import_module(module_name)
    TransformerPredictorModel = getattr(module, class_name)

    # Get Model
    model = TransformerPredictorModel(**params)

    # Get Data
    asl_dataset = ASL_DATASET(augment=True, augmentation_threshold=0.3)
    train_ds, val_ds, test_ds = create_data_loaders(asl_dataset,batch_size=BATCH_SIZE,dl_framework=DL_FRAMEWORK,
                                                               num_workers=4)
    batch = next(iter(train_ds))[0]

    # Start Training
    model(batch)

    trainer = Trainer(model, train_ds, val_ds, test_ds)

    trainer.train()
    #Todo  do we have to select the best model beforehand?
    trainer.test()

