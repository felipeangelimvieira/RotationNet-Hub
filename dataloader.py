import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def prepare_input_txt(directory, train = True):

    labels = []
    image_paths = glob.glob(directory)
    labels_file = open("labels.txt","w")
    if train:
        f = open("train_input_txt", "w")
    else:
        f = open("test_input_txt", "w")
    
    f.writelines(image_paths)
    labels_file.writelines(np.unique(labels))

class Dataloader:
    
    def __init__(self,
                 sess,
                 config,
                 shuffle_buffer_size = 1000,
                 prefetch_buffer_size = 1000,
                 image_height = 256,
                 image_width = 256,
                 seed = 42):
        tf.set_random_seed(seed)
        
        self.image_height, self.image_width = image_height, image_width
        self.sess = sess
        self.data_dir = config["data_dir"]
        self.shuffe_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.batch_size = config["batch_size"]
        self.n_views = config["n_views"]

    def get_image_path(self):
        train_dir = os.path.join(self.data_dir,'*/train/*')
        test_dir = os.path.join(self.data_dir, '*/test/*')
        
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            os.mkdir(train_dir)
            os.mkdir(test_dir)
        
        if not os.path.isfile("train_input.txt"):
            prepare_input_txt(train_dir, True)
        if not os.path.isfile("test_input.txt"):
            prepare_input_txt(test_dir, False)

    def preprocess_dataset(self, dataset):
        
        def _parse_image_from_path(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string)
            image_resized = tf.image.resize_images(image_decoded, [self.image_width,self.image_height])
            
            return image_resized

        dataset = dataset.map(_parse_image_from_path)
        dataset = dataset.batch(self.n_views)
        dataset = dataset.shuffle(self.shuffe_buffer_size)
        dataset = dataset.repeat()
        dataset.
        #TODO


        return dataset

    

    def parse_class_from_path(self,path):
        return path.replace("\\","/").split("/")[-1].split("_")[0]


    def build_pipeline(self):

        #Read file with labels for encoding
        f_labels = open("labels.txt", "r")
        self.labels = np.unique(f_labels.readlines())
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        f_labels.close()

        #Open text files with the paths to the images
        f_train = open("train_input.txt","r")
        f_test = open("test_input.txt","r")
        training_paths = f_train.readlines()
        testing_paths = f_test.readlines()

        #Create tensors
        x_train_tensor = tf.convert_to_tensor(training_paths)
        x_test_tensor = tf.convert_to_tensor(testing_paths)
        y_train_tensor = tf.convert_to_tensor(self.label_encoder.transform(map(lambda x: self.parse_class_from_path(x),training_paths)))
        y_test_tensor = tf.convert_to_tensor(self.label_encoder.transform(map(lambda x: self.parse_class_from_path(x),testing_paths)))

        #Create training and testing tf.Datasets from tensors
        x_train = tf.data.Dataset.from_tensor_slices(x_train_tensor)
        x_test = tf.data.Dataset.from_tensor_slices(x_test_tensor)
        y_train = tf.data.Dataset.from_tensor_slices(y_train_tensor)
        y_test = tf.data.Dataset.from_tensor_slices(y_test_tensor)
        
        #Zip labels and corresponding image paths
        training_dataset = tf.data.Dataset.zip((x_train,y_train))
        testing_dataset = tf.data.Dataset.zip((x_test,y_test))

        #training iterator
        training_iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        #testing iterator
        testing_iterator = tf.data.Iterator.from_structure(testing_dataset.output_types, testing_dataset.output_shapes)

        training_dataset = self.preprocess_dataset(training_dataset)


        handle = tf.placeholder(tf.string, shape = [])
        iterator = tf.data.Iterator.from_String_handle(handle,training_dataset.output_types, training_dataset.output_shapes)
        next_element = iterator.get_next()

        training_handle = sess.run(training_iterator.string_handle())
        testing_handle = sess.run(testing_iterator.string_handle())




    