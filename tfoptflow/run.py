from __future__ import absolute_import, division, print_function
import sys
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows
from optflow import flow_to_img, flow_write_as_png
import argparse
import imageio
import numpy as np
import tensorflow as tf
from copy import deepcopy

CKPT_PATH = '../../../pwcnet/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

def main():
    # parse input and output file paths
    parser = argparse.ArgumentParser(description='Run PWCNet on video.')
    parser.add_argument('input', help='input file')
    parser.add_argument('output', help='output directory')
    args = parser.parse_args()

    # set NN test options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = CKPT_PATH
    nn_opts['batch_size'] = 1
    nn_opts['use_tf_data'] = False
    nn_opts['gpu_devices'] = ['/device:GPU:0']
    nn_opts['controller'] = '/device:GPU:0'
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    nn = ModelPWCNet(mode='test', options=nn_opts)
    x_tnsr = nn.x_tnsr
    y_hat_test_tnsr = nn.y_hat_test_tnsr
    
    # read from the input video file
    video = imageio.get_reader(args.input, 'ffmpeg')

    # init = tf.global_variables_initializer()
    # nn.sess.run(init)
    prev = None
    for i, im in enumerate(video):
        # TODO: is this division necessary?
        # im /= 255.0
        if i == 0:
            prev = im
            continue
        # run flow on prev and im
        img_pairs = np.expand_dims(np.stack((prev, im)), 0)
        feed_dict = {x_tnsr: img_pairs}
        y_hat = nn.sess.run(y_hat_test_tnsr, feed_dict=feed_dict)
        pred_flows, _ = self.postproc_y_hat_test(y_hat)

        print('here')
        # show results
        plt = plot_img_pairs_w_flows(img_pairs, flow_preds=pred_flows)
        print('ok')
        plt.savefig('temp.png')
        return

        prev = im

if __name__ == '__main__':
    main()
