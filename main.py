from model import Model
from dataloader import Dataloader
from trainer import Trainer
import tensorflow as tf

config  = {"module_path": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2",
           "train_images_path": "modelnet40v2png_ori4/*/train/*",
           "test_images_path": "modelnet40v2png_ori4/*/test/*",
           "batch_size": 3,
           "n_views": 20,
           "epochs": 3,
           "v_cands": "vcand_case2.npy",
           "weight_decay": 1e-4,
           "momentum": 0.9,
           "decay_rate": 0.1,
           "decay_steps":  400e4,
           "learning_rate": 0.005 }

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
dataloader = Dataloader(sess, config)
model = Model(sess, dataloader, config)
model.build_model()
trainer = Trainer(sess,model,dataloader,config)
trainer.train()