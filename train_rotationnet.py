import argparse
import tensorflow as tf
from src.model import Model
from src.dataloader import Dataloader
from src.logger import Logger
from src.trainer import Trainer

argparser = argparse.ArgumentParser()
argparser.add_argument("--module_path", type = str, help = "Path to tf Hub module",
                        default= "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2")
argparser.add_argument("--train_images_path", type = str, help = "Path to training data",
                        default= "modelnet40v2png_ori4/*/train/*")
argparser.add_argument("--test_images_path", type = str, help = "Path to testing data",
                        default= "modelnet40v2png_ori4/*/test/*")
argparser.add_argument("--image_height", type = int, help = "Image height",
                        default= 224)
argparser.add_argument("--image_width", type = int, help = "Image width",
                        default= 224)
argparser.add_argument("--batch_size", type = int, help = "Number of objects per batch",
                        default= 3)
argparser.add_argument("--n_views", type = int, help = "Number of views per object",
                        default= 20)
argparser.add_argument("--warmup_epochs", type = float, help = "Number of epochs training only the last dense layer of RotationNet",
                        default = 5)
argparser.add_argument("--warmup_learning_rate", type = float, help = "Learning rate during warmup epochs",
                        default= 0.01)
argparser.add_argument("--epochs", type = int, help = "Number of total training epochs",
                        default = 30)
argparser.add_argument("--learning_rate", type = float, help = "Learning rate",
                        default = 0.0001)
argparser.add_argument("--decay_steps", type = int, help = "Number of steps to decay the learning rate value",
                        default= 15000)
argparser.add_argument("--decay_rate", type= float, help = "Learning rate decaying rate",
                        default= 0.9)
argparser.add_argument("--momentum", type = float, help = "Momentum",
                        default= 0.9)
argparser.add_argument("--weight_decay", type = float, help= "L2 regularization rate",
                        default = 1e-4)
argparser.add_argument("--checkpoint_dir", type = str, help = "Path for saving tensorflow checkpoints",
                        default= "checkpoints/")
argparser.add_argument("--summary_dir", type = str, help= "Path for saving summaries for tensorboard",
                        default = "summary/")
argparser.add_argument("--max_to_keep", type = int, help = "Number of checkpoints to keep",
                        default = 1)
argparser.add_argument("--v_cands", type = str, help = "Path to .npy array file with view candidates",
                        default = "vcand_case2.npy")




def main(config):

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config_proto)
    dataloader = Dataloader(sess, config, prefetch_buffer_size = 10)
    model = Model(sess, dataloader, config)
    model.build_model()
    logger = Logger(sess, config)
    trainer = Trainer(sess,model,dataloader, logger, config)
    trainer.train()



if __name__ == "__main__":
    config = argparser.parse_args()

    main(config)