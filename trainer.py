import tensorflow as tf 
import numpy as np
from tqdm import tqdm

class Trainer:

    def __init__(self,sess,model,data,config):
        self.model = model
        self.data = data
        self.sess = sess
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]


    def train(self):
        

        training_loss = []
        training_predictions = []
        training_groundtruth = []
        learning_rate = []
        
        
        iter_per_epoch = int(self.data.training_set_size/self.data.n_views/self.batch_size)
        try:
            assert(self.data.training_set_size/self.data.n_views/self.batch_size == iter_per_epoch)
        except:
            print(self.data.training_set_size)
        self.sess.run(self.data.training_initializer)
        for _ in tqdm(range(self.epochs)):
#            self.sess.run(self.data.training_initializer)
            for step in range(iter_per_epoch):
                feed_dict = {self.data.handle : self.data.training_handle}
                train_step, loss, y_true, y_pred = self.sess.run([self.model.train_step, self.model.loss, self.model.labels, self.model.predictions],
                                                            feed_dict= feed_dict)
                training_loss.append(loss)
                training_predictions.extend(y_pred)
                training_groundtruth.extend(y_true)

            training_predictions = np.array(training_predictions)
            training_groundtruth = np.array(training_groundtruth)
            print("Score for",len(training_predictions))
            print( np.mean(training_predictions == training_groundtruth))


