import tensorflow as tf

# Define a callback to save the model
class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_path, save_best_only=True,
                 monitor = "val_accuracy",
                 mode = "max"):
        super(SaveModelCallback, self).__init__()
        self.model_path = model_path
        self.save_best_only = save_best_only
        self.best_metric = float('inf')

        self.monitor = monitor
        self.mode = mode


    def on_epoch_end(self, epoch, logs=None):
        # Check if the current model is better than the previous best
        current_metric = logs.get(self.monitor)
        if self.mode == "min":
            decision = current_metric < self.best_metric
        else:
            decision = current_metric > self.best_metric

        if decision:
            self.best_metric = current_metric

            self.model.save(self.model_path)
            print(f"Saved model as loss is better than previous {self.monitor}.")
        elif not self.save_best_only:
            self.model.save(self.model_path)