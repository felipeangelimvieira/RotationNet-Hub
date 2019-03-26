import tensorflow as tf 
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support 

class Trainer:

    def __init__(self,sess,model,data, config):
        self.model = model
        self.data = data
        self.sess = sess
        self.epochs = config["epochs"]
        self.warmup_epochs = config["warmup_epochs"]
        self.batch_size = config["batch_size"]
        self.iter_per_epoch = int(self.data.training_set_size/self.data.n_views/self.batch_size)
        self.iter_per_epoch_test = int(self.data.testing_set_size/self.data.n_views/self.batch_size)
        self.warmup_steps = self.warmup_epochs*self.iter_per_epoch

    def train(self):
        
        # Initialize training handler
        self.train_op = self.model.train_step_warmup
        self.sess.run(self.data.training_initializer)
        for _ in range(self.epochs):
            self.train_epoch()
            self.test()
            

    def train_step(self):
        feed_dict = {self.data.handle : self.data.training_handle}
        _, global_step, loss, y_true, y_pred = self.sess.run([self.train_op, self.model.global_step_tensor, self.model.loss, self.model.labels, self.model.predictions],
                                                    feed_dict= feed_dict)
        return global_step, loss, y_true, y_pred

    def test_step(self):
        feed_dict = {self.data.handle : self.data.testing_handle}
        loss, y_true, y_pred = self.sess.run([self.model.loss, self.model.labels, self.model.predictions],
                                                    feed_dict= feed_dict)
        return loss, y_true, y_pred

    def compute_precision_recall_fscore(self, y_true, y_pred):
        y_true = np.array(y_true).reshape((-1,))
        y_pred = np.array(y_pred).reshape((-1,))
        precision, recall, f1, support =  precision_recall_fscore_support(y_true, y_pred)
        return precision.mean(), recall.mean(), f1.mean()

    def train_epoch(self):
        """
        Train one epoch
        """

        #keep track of training loss
        training_loss = []
        training_predictions = []
        training_groundtruth = []
        learning_rate = []
        
        # Initialize training handler
        self.sess.run(self.data.training_initializer)

        training_steps = tqdm(range(self.iter_per_epoch))
        for step in training_steps:

            global_step, loss, y_true, y_pred = self.train_step()
            if global_step > self.warmup_steps:
                self.train_op = self.model.train_step
            
            training_loss.append(loss)
            training_predictions.extend(y_pred)
            training_groundtruth.extend(y_true)
            training_steps.set_description("Loss = " + str(loss))

        training_precision, training_recall, training_f1 = self.compute_precision_recall_fscore(training_groundtruth, training_predictions)
        print(training_precision, training_recall, training_f1)

    def test(self):
        
        testing_loss = []
        testing_predictions = []
        testing_groundtruth = []
        # Initialize testing dataset
        self.sess.run(self.data.testing_initializer)
        for step in tqdm(range(self.iter_per_epoch_test)):
            loss, y_true, y_pred = self.test_step()
            testing_loss.append(loss)
            testing_predictions.extend(y_pred)
            testing_groundtruth.extend(y_true)

        testing_precision, testing_recall, testing_f1 =  self.compute_precision_recall_fscore(testing_groundtruth, testing_predictions)
        print(testing_precision, testing_recall, testing_f1)