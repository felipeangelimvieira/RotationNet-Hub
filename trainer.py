import tensorflow as tf 
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support 

class Trainer:

    def __init__(self,sess,model,data,config):
        self.model = model
        self.data = data
        self.sess = sess
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]


    def train(self):
        

        
        iter_per_epoch = int(self.data.training_set_size/self.data.n_views/self.batch_size)
        iter_per_epoch_test = int(self.data.testing_set_size/self.data.n_views/self.batch_size)

        print(iter_per_epoch)
        print(self.data.training_set_size)
        print(self.batch_size)
        print(self.data.n_views)
        try:
            assert(self.data.training_set_size/self.data.n_views/self.batch_size == iter_per_epoch)
        except:
            print(self.data.training_set_size)

        # Initialize training handler
        self.sess.run(self.data.training_initializer)
        for _ in range(self.epochs):
            training_loss = []
            training_predictions = []
            training_groundtruth = []
            learning_rate = []
            
            # Initialize training handler
            self.sess.run(self.data.training_initializer)
            for step in tqdm(range(iter_per_epoch)):
                feed_dict = {self.data.handle : self.data.training_handle}
                train_step, loss, y_true, y_pred = self.sess.run([self.model.train_step, self.model.loss, self.model.labels, self.model.predictions],
                                                            feed_dict= feed_dict)
                training_loss.append(loss)
                training_predictions.extend(y_pred)
                training_groundtruth.extend(y_true)

                
            training_predictions = np.array(training_predictions)
            training_groundtruth = np.array(training_groundtruth)
            training_precision, training_recall, training_f1, training_support =  precision_recall_fscore_support(training_groundtruth, training_predictions)
            
            
            testing_loss = []
            testing_predictions = []
            testing_groundtruth = []
            # Initialize testing dataset
            self.sess.run(self.data.testing_initializer)
            for step in range(iter_per_epoch_test):
                feed_dict = {self.data.handle : self.data.testing_handle}
                loss, y_true, y_pred = self.sess.run([self.model.loss, self.model.labels, self.model.predictions],
                                                            feed_dict= feed_dict)
                testing_loss.append(loss)
                testing_predictions.extend(y_pred)
                testing_groundtruth.extend(y_true)

            testing_predictions = np.array(testing_predictions)
            testing_groundtruth = np.array(testing_groundtruth)
            testing_precision, testing_recall, testing_f1, testing_support =  precision_recall_fscore_support(testing_groundtruth, testing_predictions)


            # Summarize
            tf.summary.scalar("Training precision", training_precision)
            tf.summary.scalar("Training recall", training_recall)
            tf.summary.scalar("Training f1 score", training_f1)
            tf.summary.scalar("Training loss", training_loss.mean())
            tf.summary.scalar("Testing precision", testing_precision)
            tf.summary.scalar("Testing recall", testing_recall)
            tf.summary.scalar("Testing f1 score", testing_f1)
            tf.summary.scalar("Testing loss", testing_loss.mean())
            

