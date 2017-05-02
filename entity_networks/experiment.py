from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import argparse
import tensorflow as tf

from entity_networks.inputs import generate_input_fn
from entity_networks.serving import generate_serving_input_fn
from entity_networks.model import model_fn

BATCH_SIZE = 32
NUM_BLOCKS = 20
EMBEDDING_SIZE = 100
CLIP_GRADIENTS = 40.0

def generate_experiment_fn(data_dir, dataset_id, num_epochs,
                           learning_rate_min, learning_rate_max,
                           learning_rate_step_size):
    "Return _experiment_fn for use with learn_runner."
    def _experiment_fn(output_dir):
        metadata_path = os.path.join(data_dir, '{}_10k.json'.format(dataset_id))
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

        train_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'train'))
        eval_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'test'))

        train_input_fn = generate_input_fn(
            filename=train_filename,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=num_epochs,
            shuffle=True)

        eval_input_fn = generate_input_fn(
            filename=eval_filename,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        run_config = tf.contrib.learn.RunConfig()

        vocab_size = metadata['vocab_size']
        task_size = metadata['task_size']
        train_steps_per_epoch = task_size // BATCH_SIZE

        params = {
            'vocab_size': vocab_size,
            'embedding_size': EMBEDDING_SIZE,
            'num_blocks': NUM_BLOCKS,
            'learning_rate_min': learning_rate_min,
            'learning_rate_max': learning_rate_max,
            'learning_rate_step_size': learning_rate_step_size * train_steps_per_epoch,
            'clip_gradients': CLIP_GRADIENTS,
        }
        print(params)

        estimator = tf.contrib.learn.Estimator(
            model_dir=output_dir,
            model_fn=model_fn,
            config=run_config,
            params=params)

        eval_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy)
        }

        serving_input_fn = generate_serving_input_fn(metadata)
        export_strategy = tf.contrib.learn.make_export_strategy(
            serving_input_fn=serving_input_fn)

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_steps=None,
            eval_steps=None,
            export_strategies=[export_strategy])
        return experiment

    return _experiment_fn
