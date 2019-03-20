import os
import numpy as np
import tensorflow as tf
import glob
from sklearn.preprocessing import LabelEncoder

def prepare_input_txt(directory, train = True):

    labels = []
    image_paths = glob.glob(directory)
    print(len(image_paths))
    image_paths = list(map(lambda s: os.path.abspath(s).replace("\\","/"), image_paths))
    if train:
        f = open("train_input.txt", "w")
    else:
        f = open("test_input.txt", "w")
    
    f.write("\n".join(image_paths))
    f.close()

class Dataloader:
    
    def __init__(self,
                 sess,
                 config,
                 shuffle_buffer_size = 10,
                 prefetch_buffer_size = 10,
                 image_height = 224,
                 image_width = 224,
                 seed = 42):
        tf.set_random_seed(seed)
        
        self.image_height, self.image_width = image_height, image_width
        self.sess = sess
        #self.data_dir = config["data_dir"]
        self.train_images_path = config["train_images_path"]
        self.test_images_path = config["test_images_path"]
        self.shuffe_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.batch_size = config["batch_size"]
        self.n_objects = config["batch_size"]
        self.n_views = config["n_views"]
        self.n_images = self.n_objects*self.n_views
        self.epochs = config["epochs"]

        self.get_image_path()
        self.build_pipeline()

    def get_image_path(self):
        #train_dir = os.path.join(self.data_dir,'*/train/*')
        #test_dir = os.path.join(self.data_dir, '*/test/*')
    

        #if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        #    os.mkdir(train_dir)
        #    os.mkdir(test_dir)
        
        if not os.path.isfile("train_input.txt"):
            prepare_input_txt(self.train_images_path, True)
        if not os.path.isfile("test_input.txt"):
            prepare_input_txt(self.test_images_path, False)

    def preprocess_dataset(self, dataset):
        
        def _parse_image_from_path(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string)
            image_resized = tf.image.resize_images(image_decoded, [self.image_width,self.image_height])
            
            return image_resized, label

        dataset = dataset.map(_parse_image_from_path)
        dataset = dataset.batch(self.n_views)
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.shuffle(self.shuffe_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        #TODO
        return dataset

    

    def parse_class_from_path(self,path):
        parsed = path.replace("\\","/").split("/")[-1].split("_")[:-2]
        return "_".join(parsed)

    def build_pipeline(self):

        #Read file with labels for encoding
        f_labels = open("labels.txt", "r")
        self.labels = np.unique(f_labels.read().splitlines())
        print(self.labels)
        self.n_classes = len(self.labels)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        f_labels.close()

        #Open text files with the paths to the images
        f_train = open("train_input.txt","r")
        f_test = open("test_input.txt","r")
        training_paths = f_train.read().splitlines()
        testing_paths = f_test.read().splitlines()

        self.training_set_size = len(training_paths)
        #Create tensors
        x_train_tensor = tf.convert_to_tensor(training_paths)
        x_test_tensor = tf.convert_to_tensor(testing_paths)
        y_train_tensor = tf.convert_to_tensor(self.label_encoder.transform(list(map(lambda x: self.parse_class_from_path(x),training_paths))))
        y_test_tensor = tf.convert_to_tensor(self.label_encoder.transform(list(map(lambda x: self.parse_class_from_path(x),testing_paths))))

        #Create training and testing tf.Datasets from tensors
        x_train = tf.data.Dataset.from_tensor_slices(x_train_tensor)
        x_test = tf.data.Dataset.from_tensor_slices(x_test_tensor)
        y_train = tf.data.Dataset.from_tensor_slices(y_train_tensor)
        y_test = tf.data.Dataset.from_tensor_slices(y_test_tensor)
        
        #Zip labels and corresponding image paths
        training_dataset = tf.data.Dataset.zip((x_train,y_train))
        testing_dataset = tf.data.Dataset.zip((x_test,y_test))

        
        training_dataset = self.preprocess_dataset(training_dataset)
        testing_dataset = self.preprocess_dataset(testing_dataset)

        #training iterator
        training_iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        #testing iterator
        testing_iterator = tf.data.Iterator.from_structure(testing_dataset.output_types, testing_dataset.output_shapes)



        self.handle = tf.placeholder(tf.string, shape = [])
        iterator = tf.data.Iterator.from_string_handle(self.handle,training_dataset.output_types, training_dataset.output_shapes)
        self.next_element = iterator.get_next() 

        self.training_handle = self.sess.run(training_iterator.string_handle())
        self.testing_handle = self.sess.run(testing_iterator.string_handle())

        self.training_initializer = training_iterator.make_initializer(training_dataset)
        self.testing_initializer = testing_iterator.make_initializer(testing_dataset)

        return self.next_element




    