import sys

sys.path.insert(0, '../src')

from config import *
from src.data.dataset import ASL_DATASET
from src.data.data_utils import create_data_loaders

import time
import gc
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

import torch
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=EPOCHS,
                save_freq=5, verbose=True):

    """
    Cooments
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param save_freq:
    :param verbose:
    :return:
    """
    print(f"Training started")
    print(f"    Mode          : {DEVICE}")
    print(f"    Model type    : {type(model)}")

    start_time = time.time()

    # Initialize lists for tracking losses and ROCs
    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = float('inf'), 0.0

    # TODO
    best_model_path = ''  # output at config.OUT_DIR
    best_epoch = 0

    model.zero_grad()
    model.to(DEVICE)

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=os.path.join(ROOT_PATH, RUNS_DIR))

    for epoch in range(1, num_epochs + 1):
        model.train()
        print(f"Epoch {epoch}")

        running_loss = 0.0
        running_accuracy = 0.0
        
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs; data is a dictionary like {'landmarks': landmarks, 'target': target, 'size': size}
            inputs, labels, seq_lengths = data['landmarks'], data['target'], data['size']

            # Move data to the device
            inputs, labels, seq_lengths = inputs.to(DEVICE), labels.to(DEVICE), seq_lengths.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, seq_lengths)

            if type(outputs) == tuple and len(outputs) == 2:
                outputs = outputs[0]

            outputs.to(DEVICE)

            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()
            running_accuracy += acc

            # Clean up GPU memory if needed
            if DEVICE == 'cuda':
                del inputs, labels, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()

            # print statistics
            if verbose:
                if (i + 1) % 50 == 0:  # print every 50 mini-batches
                    print(f' Epoch: {epoch:>2} '
                          f' Batch: {i + 1:>3} / {len(train_loader)} '
                          f' loss: {running_loss / (i + 1):>6.4f} '
                          f' accuracy: {running_accuracy / (i + 1):>6.4f} '
                          f' Average batch time: {(time.time() - epoch_start_time) / (i + 1):>4.3f} secs')

        print(f"\nTraining loss: {running_loss / len(train_loader):>4f}")
        print(f"Training accuracy: {running_accuracy / len(train_loader):>4f}") 
    
        train_losses.append(running_loss / len(train_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)  # write loss to TensorBoard
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % 5 == 0:  # save model and checkpoint for inference or training
            save_model(model, epoch)
        
        # validation
        print("Validating...")
        model.eval()
        
        val_loss, val_accuracy = 0.0, 0.0

        with torch.no_grad():
            for data in valid_loader:
                inputs, labels, seq_lengths = data['landmarks'], data['target'], data['size']
                inputs, labels, seq_lengths = inputs.to(DEVICE), labels.to(DEVICE), seq_lengths.to(DEVICE)

                outputs = model(inputs, seq_lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += acc

        val_loss /= len(valid_loader)
        val_accuracy /= len(valid_loader)
        
        val_losses.append(val_loss)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        # Step the learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        writer.flush()

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best validation loss: {best_val_loss:.3f}, achieved on epoch #{best_epoch}")

    writer.close()

    return best_model_path, train_losses, val_losses, val_rocs
                           
                           
def save_model(model, num_epochs):
    model_subdir = 'model_' + str(model)
    folder_path = os.path.join(ROOT_PATH, MODEL_DIR, model_subdir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, f'{str(model)}_{num_epochs}.pth')
    torch.save(model.state_dict(), path)
    return path                        

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def train():
    asl_dataset = ASL_DATASET(augment=True)

    train_loader, valid_loader, test_loader = create_data_loaders(asl_dataset)

    from models.models import ImprovedLSTMModel

    model = ImprovedLSTMModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=EPOCHS,
                save_freq=5, verbose=True)

if __name__ == '__main__':
    train()