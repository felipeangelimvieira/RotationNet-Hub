from model import Model
from dataloader import Dataloader
import tensorflow as tf

config  = {"batch_size": 3,
           "n_views": 12,
           "module_path": ,
           "n_epochs": 20,
           "v_cands": ,
           "weight_decay": ,
           "decay_rate": ,
           "decay_steps": ,
           "learning_rate": }

- batch_size: number of objects per batch
- n_views
- module_path
- v_cands (path)
- n_epochs
- weight_decay
- decay_rate
- decay_steps
- learning_rate

sess = tf.Session()
dataloader = Dataloader(sess, config)
model = Model(dataloader, config)
model.build_model()