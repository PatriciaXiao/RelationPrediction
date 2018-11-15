# Graph Convolutional Networks for Relational Link Prediction

This repository contains a TensorFlow implementation of Relational Graph Convolutional Networks (R-GCN), as well as experiments on relational link prediction. The description of the model and the results can be found in out paper:

[Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling (ArXiv 2017)

**Requirements**

* TensorFlow (>1.0)

**Running demo**

We provide a bash script to run a demo of our code. In the folder *settings*, a collection of configuration files can be found. The block diagonal model used in our paper is represented through the configuration file *settings/gcn_block.exp*. To run a given experiment, execute our bash script as follows:

```
bash run-train.sh \[configuration\]
```

We advise running the model on a modern GPU, as training can take up to several hours.

**Citation**

Please cite our paper if you use this code in your own work:

```
@article{schlichtkrull2017modeling,
  title={Modeling Relational Data with Graph Convolutional Networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  journal={arXiv preprint arXiv:1703.06103},
  year={2017}
}
```

## My notes

```
virtualenv -p python3 venv
source ./venv/bin/activate
pip install theano
pip install tensorflow

cd code
python train.py --settings ../settings/distmult.exp --dataset ../data/Toy
python train.py --settings ../settings/distmult.exp --dataset ../data/twitter
python train.py --settings ../settings/distmult.exp --dataset ../data/twitter_mini
```

Receives warning:
```
WARNING:tensorflow:From /Users/patricia.xiao/Documents/RelationPrediction/venv/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:189: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
```
in the training phase.