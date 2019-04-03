import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class Model:
    def __init__(self, sess, data, config, learning_rate = 0.05, decay_rate = 0.9, decay_steps = 1e100, weight_decay = 0.005, load = True):
        self.data = data
        self.sess = sess
        self.n_classes = data.n_classes
        self.n_objects = data.n_objects
        self.n_views = config["n_views"]
        self.module_path = config["module_path"]
        self.v_cands = np.load(config["v_cands"])
        self.batch_size = config["batch_size"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.n_images = self.batch_size*self.n_views
        self.indexes = self.indexes_to_gather(self.v_cands,self.n_objects)
        self.n_cands = self.v_cands.shape[0]

        self.warmup_learning_rate = config["warmup_learning_rate"]
        self.warmup_epochs = config["warmup_epochs"]
        self.weight_decay = config["weight_decay"]
        self.decay_steps = config["decay_steps"]
        self.decay_rate = config["decay_rate"]
        self.learning_rate = config["learning_rate"]
        self.momentum = config["momentum"]
        self.max_to_keep = config["max_to_keep"]
        self.init_global_step()
        self.init_saver()
        if load:
            self.load(self.sess)
        

    def indexes_to_gather(self,v_cands,n_objects):
        
        # n_objects,n_views_per_obj,n_views,n_classes+1
        
        candidate_indexes = []
        
        for obj in range(n_objects):
            obj_cand = []
            for i in range(v_cands.shape[0]):
                v_cand = []
                for j in range(v_cands.shape[1]):
                    v_cand.append([obj,j,v_cands[i,j]])
                obj_cand.append(v_cand)
            candidate_indexes.append(obj_cand)
            
        #tensor of shape [n_objs, n_cands, n_views,3]
        return tf.convert_to_tensor(candidate_indexes, tf.int64)
    
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.sess.run(self.global_step_tensor.initializer)
        
        
    def build_model(self):

        
        module = hub.Module(self.module_path, trainable= True)
        self.height, self.width =  hub.get_expected_image_size(module)
        print(self.height, self.width, "AAAA")
        self.input, self.labels = self.data.next_element 
        
        print(self.input)
        print(self.labels)
        self.input = tf.reshape(self.input, [-1, self.height, self.width, 3])
        self.labels = tf.reshape(self.labels, [-1])
        
        
        self.hidden_layer = module(self.input)
        self.logits = tf.layers.dense(self.hidden_layer, (self.n_classes+1)*self.n_views, activation= None, name = "logits")    
        
        #n_objects_per_batch,n_views_in_batch,n_views_logits,n_classes+1
        self.probs = tf.nn.softmax(tf.reshape(self.logits, [self.n_objects,self.n_views,self.n_views,self.n_classes+1]), axis = -1)
        self.log_p = tf.math.log(self.probs)
        
        self.scores = self.log_p[...,:-1] - tf.tile(self.log_p[...,-1:],[1,1,1,self.n_classes])
        
        """
        Calculate the score for each view order candidate
        Determine best view order candidate
        
        Create a gathering tensor to hook the scores for groundtruth label y
        Last dimension of the gathering tensor is [object index, input image index, view candidate, label]
        
        We gather these values for every candidate, and create a matrix C of shape [n_objects,n_cands]
        where C[i,j] is the final score for object i and view order candidate j
        
        Finally, we gather the best indexes in gathering tensor for calculating the loss
        """
        tiled = tf.tile(tf.reshape(self.labels,[self.n_objects,-1]),[1,self.n_cands])
        tiled = tf.reshape(tiled,[self.n_objects,self.n_cands,self.n_views, 1])
        #tensor of shape [n_objs, n_cands, n_views,4]
        self.gather_candidate_scores = tf.concat([self.indexes,tiled], axis = -1)
        # candidates[i,j] is the score for object i and view order candidate j
        self.candidate_scores = tf.reduce_sum(tf.gather_nd(self.scores,self.gather_candidate_scores), axis = -1)
        
        best_candidates = tf.reshape(tf.argmin(self.candidate_scores, -1),[self.n_objects,1])
        # pair [[0,cand_0],[1,cand_1],...]
        best_candidates = tf.concat([tf.reshape(tf.range(0,self.n_objects, dtype = tf.int64),[self.n_objects,1]),best_candidates], axis = -1)
        
        """
        Calculate loss considering best order candidate
        """
        
        
        var_list = tf.trainable_variables()[-2:]
        # Train op
        with tf.name_scope("train"):
            #loss function
            with tf.name_scope("loss"):
                self.labels = tf.reshape(self.labels,[-1,self.n_views])[:,0]
                #Indexes to calculate cross-entropy loss of best view candidates
                #shape [n_objects,n_views,3]
                self.gather_candidate_log_prob = tf.gather_nd(self.gather_candidate_scores,best_candidates)
                
                #Indexes to dum the loss of i_view
                #Discount the loss of i_view for the best view point
                discount_iview = tf.concat([(self.n_classes+1)*tf.ones([self.n_objects, self.n_views,1], dtype = tf.int64),self.gather_candidate_log_prob[...,:-1]], axis = -1)
                self.discount_iview = discount_iview
               
                self.loss = -tf.gather_nd(self.log_p,self.gather_candidate_log_prob)
                self.loss = self.loss -  tf.reduce_mean(self.log_p[:,:,:,-1], axis = -1)
                self.loss = self.loss + tf.gather_nd(self.log_p,discount_iview)
                self.loss = tf.reduce_mean(self.loss)

                #l2 loss (improves the performance)
                for var in var_list:
                    self.loss += tf.nn.l2_loss(var)*self.weight_decay
                    
                self.predictions = self.select_best(self.logits)
                
            

            
            # setting different training ops for each part of the network
            # Get gradients of all trainable variables
            gradients = tf.gradients(self.loss, var_list)

            optimizer = tf.train.MomentumOptimizer(self.warmup_learning_rate,self.momentum)
            training_op = optimizer.apply_gradients(zip(gradients, var_list), global_step = self.global_step_tensor)
            self.train_step_warmup = training_op

            var_list = tf.trainable_variables()
            # learning rate
            self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                            self.global_step_tensor,
                                                            self.decay_steps,
                                                            self.decay_rate,
                                                            staircase=True)
            
            gradients = tf.gradients(self.loss, var_list)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
            training_op = optimizer.apply_gradients(zip(gradients, var_list), global_step = self.global_step_tensor)
            self.train_step = training_op
            self.sess.run(tf.global_variables_initializer())

            
    def select_best(self,fc8):
        #shape batch_size x P matrix
        fc8 =  tf.reshape(fc8,[-1, self.n_views, self.n_classes + 1])
        #apply softmax on the rows of the P matrix
        fc8 = tf.log(tf.nn.softmax(fc8,axis = -1))
        self.softmax_fc8 = fc8
        #divide all probs by the probability of incorrect view (shape (batch_size,view_num,classes_num))
        i_view_probability_to_sub = tf.tile(tf.expand_dims(fc8[...,-1],-1),[1,1,self.n_classes])
        fc8 = fc8[...,:-1]
        score = fc8 - i_view_probability_to_sub
        self.score1 = score
        # score per image per object
        score =  tf.reshape(score,[self.n_objects,self.n_views, self.n_views, self.n_classes])
        self.score = tf.reduce_sum(tf.reduce_max(score, axis = -2),axis = -2)
        best_classes = tf.argmax(self.score,axis = -1)
        return tf.cast(best_classes,dtype = tf.int64)

    def init_saver(self):
        self.saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = self.max_to_keep)

     # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        print("Latest checkpoint:",latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
