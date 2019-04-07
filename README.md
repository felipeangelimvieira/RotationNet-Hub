# RotationNet Hub
A versatile implementation of RotationNet, using models from Tensorflow Hub.

Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida.
**RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints.** 
*CVPR*, pp.5010-5019, 2018.
([pdf](https://arxiv.org/abs/1603.06208))
([project](https://kanezaki.github.io/rotationnet/))


## Dependencies
- tensorflow >= 1.13.0
- numpy
- tqdm

## Usage

### From command line

Call train_rotationnet.py file.

```bash
python train_rotationnet.py [-h] [--module_path MODULE_PATH]
                            [--train_images_path TRAIN_IMAGES_PATH]
                            [--test_images_path TEST_IMAGES_PATH]
                            [--image_height IMAGE_HEIGHT]
                            [--image_width IMAGE_WIDTH]
                            [--batch_size BATCH_SIZE] [--n_views N_VIEWS]
                            [--warmup_epochs WARMUP_EPOCHS]
                            [--warmup_learning_rate WARMUP_LEARNING_RATE]
                            [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                            [--decay_steps DECAY_STEPS]
                            [--decay_rate DECAY_RATE] [--momentum MOMENTUM]
                            [--weight_decay WEIGHT_DECAY]
                            [--checkpoint_dir CHECKPOINT_DIR]
                            [--summary_dir SUMMARY_DIR]
                            [--max_to_keep MAX_TO_KEEP] [--v_cands V_CANDS]

optional arguments:
  -h, --help            show this help message and exit
  --module_path MODULE_PATH
                        Path to tf Hub module
  --train_images_path TRAIN_IMAGES_PATH
                        Path to training data
  --test_images_path TEST_IMAGES_PATH
                        Path to testing data
  --image_height IMAGE_HEIGHT
                        Image height
  --image_width IMAGE_WIDTH
                        Image width
  --batch_size BATCH_SIZE
                        Number of objects per batch
  --n_views N_VIEWS     Number of views per object
  --warmup_epochs WARMUP_EPOCHS
                        Number of epochs training only the last dense layer of
                        RotationNet
  --warmup_learning_rate WARMUP_LEARNING_RATE
                        Learning rate during warmup epochs
  --epochs EPOCHS       Number of total training epochs
  --learning_rate LEARNING_RATE
                        Learning rate
  --decay_steps DECAY_STEPS
                        Number of steps to decay the learning rate value
  --decay_rate DECAY_RATE
                        Learning rate decaying rate
  --momentum MOMENTUM   Momentum
  --weight_decay WEIGHT_DECAY
                        L2 regularization rate
  --checkpoint_dir CHECKPOINT_DIR
                        Path for saving tensorflow checkpoints
  --summary_dir SUMMARY_DIR
                        Path for saving summaries for tensorboard
  --max_to_keep MAX_TO_KEEP
                        Number of checkpoints to keep
  --v_cands V_CANDS     Path to .npy array file with view candidates
```

### Jupyter notebook

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
