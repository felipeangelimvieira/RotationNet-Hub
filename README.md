# RotationNet Hub
A versatile implementation of RotationNet, using models from Tensorflow Hub.

Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida.
**RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints.** 
*CVPR*, pp.5010-5019, 2018.
([pdf](https://arxiv.org/abs/1603.06208))
([project](https://kanezaki.github.io/rotationnet/))

## Dependencies
- tensorflow >= 1.13.0
- tqdm

## Usage

```python

from src.model import Model
from src.trainer import Trainer
from src.dataloader import Dataloader
from src.logger import Logger
from src.utils import Config

config  = {"module_path": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2",
           "train_images_path": "modelnet40v2png_ori4/*/train/*",
           "test_images_path": "modelnet40v2png_ori4/*/test/*",
           "image_height": 224,
           "image_width": 224,
           "batch_size": 3,
           "n_views": 20,
           "warmup_epochs": 3,
           "checkpoint_dir": "checkpoints/",
           "summary_dir": "summary/",
           "warmup_learning_rate": 0.01,
           "max_to_keep": 1,
           "epochs": 30,
           "v_cands": "vcand_case2.npy",
           "weight_decay": 1e-4,
           "momentum": 0.9,
           "decay_rate": 0.9,
           "decay_steps":  12000,
           "learning_rate": 0.00005 }

#Model config
config = Config(config)
#Session config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
#create session
sess = tf.InteractiveSession(config=config_proto)
#create dataloader
dataloader = Dataloader(sess, config, prefetch_buffer_size = 10)
#create the model
model = Model(sess, dataloader, config)
model.build_model()
#create a logger instance for saving summaries
logger = Logger(sess, config)
#create Trainer and train your RotationNet
trainer = Trainer(sess,model,dataloader, logger, config)
trainer.train()
```
- module_path

## Config

- batch_size: number of objects per batch
- n_views
- module_path
- v_cands (path)
- n_epochs
- weight_decay 
- decay_rate
- decay_steps
- learning_rate