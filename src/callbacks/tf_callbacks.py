import tensorflow as tf

# Define a callback to save the model
class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_path, save_best_only=True):
        super(SaveModelCallback, self).__init__()
        self.model_path = model_path
        self.save_best_only = save_best_only
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # Check if the current model is better than the previous best
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.save(self.model_path)
            print("Saved model as loss is better than previous loss.")
        elif not self.save_best_only:
            self.model.save(self.model_path)