# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

We build our resize-compress framework on top of the image compression model pub-
lished in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With the official Github repository: https://github.com/tensorflow/compression

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from modules.warp_utils import affine_grid_generator, bicubic_sampler
from modules.rc_networks import ResizeParamNet, PreSubSample, PostReSample
from modules.bls2017_networks import AnalysisTransform, SynthesisTransform


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)
  pre_subsample = PreSubSample(32)
  post_resample = PostReSample(32)
  resize_param = ResizeParamNet()

  """
  Forward bicubic sampling (downscale)
  """
  B = tf.shape(x)[0]
  H = tf.shape(x)[1]
  W = tf.shape(x)[2]
  with tf.variable_scope("loc_net"): #, reuse=tf.AUTO_REUSE):
    f = tf.cast(resize_param(x), tf.float64)
  with tf.variable_scope("warp_param"):
    f0 = tf.zeros_like(f, dtype=tf.float64)
    theta_fw = tf.concat([1/f, f0, f0, f0, 1/f, f0], axis=1)
    theta_fw = tf.reshape(theta_fw, [B, 2, 3])
    theta_inv = tf.concat([f, f0, f0, f0, f, f0], axis=1)
    theta_inv = tf.reshape(theta_inv, [B, 2, 3])

    # Pre-filtering before bicubic interpolation
    x_pre = pre_subsample(x)
    # Peform bicubic interpolation
    batch_grids = affine_grid_generator(H, W, theta_fw, B)
    x_b = bicubic_sampler(x_pre, batch_grids[:, 0, :, :], batch_grids[:, 1, :, :])

  """
  compress by BLS2017 model
  """
  # Build autoencoder.
  y = analysis_transform(x_b)

  ## Todo: clean-up this part. Currently only batchsize = 8 supported
  psize = tf.cast(tf.math.ceil(args.patchsize / 16), 'int32')
  canvas_crop_max = tf.math.minimum(psize, tf.cast(tf.math.ceil((args.patchsize / 16) * f), 'int32'))
  y0, l0 = entropy_bottleneck(y[0:1,0:canvas_crop_max[0,0],0:canvas_crop_max[0,0],:], training=True)
  y1, l1 = entropy_bottleneck(y[1:2,0:canvas_crop_max[1,0],0:canvas_crop_max[1,0],:], training=True)
  y2, l2 = entropy_bottleneck(y[2:3,0:canvas_crop_max[2,0],0:canvas_crop_max[2,0],:], training=True)
  y3, l3 = entropy_bottleneck(y[3:4,0:canvas_crop_max[3,0],0:canvas_crop_max[3,0],:], training=True)
  y4, l4 = entropy_bottleneck(y[4:5,0:canvas_crop_max[4,0],0:canvas_crop_max[4,0],:], training=True)
  y5, l5 = entropy_bottleneck(y[5:6,0:canvas_crop_max[5,0],0:canvas_crop_max[5,0],:], training=True)
  y6, l6 = entropy_bottleneck(y[6:7,0:canvas_crop_max[6,0],0:canvas_crop_max[6,0],:], training=True)
  y7, l7 = entropy_bottleneck(y[7:8,0:canvas_crop_max[7,0],0:canvas_crop_max[7,0],:], training=True)
  # y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  y0 = tf.pad(y0, [[0,0], [0,psize-canvas_crop_max[0,0]], [0,psize-canvas_crop_max[0,0]], [0,0]], 'CONSTANT', constant_values=0)
  y1 = tf.pad(y1, [[0,0], [0,psize-canvas_crop_max[1,0]], [0,psize-canvas_crop_max[1,0]], [0,0]], 'CONSTANT', constant_values=0)
  y2 = tf.pad(y2, [[0,0], [0,psize-canvas_crop_max[2,0]], [0,psize-canvas_crop_max[2,0]], [0,0]], 'CONSTANT', constant_values=0)
  y3 = tf.pad(y3, [[0,0], [0,psize-canvas_crop_max[3,0]], [0,psize-canvas_crop_max[3,0]], [0,0]], 'CONSTANT', constant_values=0)
  y4 = tf.pad(y4, [[0,0], [0,psize-canvas_crop_max[4,0]], [0,psize-canvas_crop_max[4,0]], [0,0]], 'CONSTANT', constant_values=0)
  y5 = tf.pad(y5, [[0,0], [0,psize-canvas_crop_max[5,0]], [0,psize-canvas_crop_max[5,0]], [0,0]], 'CONSTANT', constant_values=0)
  y6 = tf.pad(y6, [[0,0], [0,psize-canvas_crop_max[6,0]], [0,psize-canvas_crop_max[6,0]], [0,0]], 'CONSTANT', constant_values=0)
  y7 = tf.pad(y7, [[0,0], [0,psize-canvas_crop_max[7,0]], [0,psize-canvas_crop_max[7,0]], [0,0]], 'CONSTANT', constant_values=0)

  l0 = tf.reshape(l0,[-1])
  l1 = tf.reshape(l1,[-1])
  l2 = tf.reshape(l2,[-1])
  l3 = tf.reshape(l3,[-1])
  l4 = tf.reshape(l4,[-1])
  l5 = tf.reshape(l5,[-1])
  l6 = tf.reshape(l6,[-1])
  l7 = tf.reshape(l7,[-1])

  y_tilde = tf.concat([y0, y1, y2, y3, y4, y5, y6, y7], 0)
  likelihoods = tf.concat([l0, l1, l2, l3, l4, l5, l6, l7], 0)

  # Forming y_tilde for inverse transform
  x_tilde_b = synthesis_transform(y_tilde)

  """
  Backward bicubic sampling (upscale)
  """
  with tf.variable_scope("warp_param"):
    # Peform bicubic interpolation
    batch_grids2 = affine_grid_generator(H, W, theta_inv, B)
    x_tilde_f = bicubic_sampler(x_tilde_b, batch_grids2[:, 0, :, :], batch_grids2[:, 1, :, :])
    # Post-filtering after bicubic interpolation
    x_tilde = post_resample(x_tilde_f)

  """
  Define loss functions
  """
  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
  

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Define trainable variables for compression
  f_vars = [var for var in tf.trainable_variables() if 'warp_param' in var.name]
  loc_vars = [var for var in tf.trainable_variables() if 'loc_net' in var.name]
  bls_vars = [var for var in tf.trainable_variables() if 'loc_net' not in var.name]
  for var in f_vars: print(var.name)
  for var in loc_vars: print(var.name)
  for var in bls_vars: print(var.name)

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step, var_list=bls_vars)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  f_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  f_step = f_optimizer.minimize(train_loss, var_list=loc_vars)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # These are needed for updating
  with tf.control_dependencies(update_ops):               # moving mean / variance of BNs
    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0], f_step)

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  tf.summary.image("fw_warp", quantize_image(x_b))
  tf.summary.image("bw_warp", quantize_image(x_tilde_b))

  tf.summary.histogram("resize_factor", f)
  tf.summary.scalar("E[resize_factor]", tf.reduce_mean(f))

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def parse_args(argv):
  print(argv)
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'compress' subcommand.
  compress_patch_cmd = subparsers.add_parser(
      "compress_patch",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (compress_patch_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))
    cmd.add_argument(
        "W",
        help="input width.")
    cmd.add_argument(
        "H",
        help="input height.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    assert False, "Test code (compress) not implemented yet"
  elif args.command == "decompress":
    assert False, "Test code (decompress) not implemented yet"


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
