# NTU MLDS2017 final part1
**Team: 培傑與順耀**

**Team Members:B03901056 孫凡耕, B03901032 郭子生, B03901003 許晉嘉, B03901070 羅啟心**

## Followings should be installed and versions suggested:
- tensorflow-gpu >= 1.0.0
- python-package : tensorpack >= 0.2.0

## Usage
- `python3 test_[alexnet, googlenet, vgg16, vgg19].py -h` to see help message
- In summary, need to provide:
  1. one input images
  2. method to visualize (0->deconvolution, 1->activation, 2->saliency_map)
  3. layers to be visualize ('r'->relu, 'p'->pooling, 'c'->convolution)
  4. prefix of log directory and output directory

  p.s.1 currently `AlexNet` is the only one that supports `saliency_map`

  p.s.2 Need to run `deconvolution` or `activation` once before running `saliency_map`

  Ex. `python3 testAlexNet.py -i imgs/car.jpg -m 0 -p Deconv_Car -l rpc`
- If `deconvolution` and `activation` are chosen, open tensorboard and choose the
  corresponding logdir to visualize images.
  If `saliency_map` are chosen, prefix_pos.jpg, prefix_neg.jpg, prefix_abs.jpg
  and prefix_blended.jpg will be produced in the same directory. "pos" and "neg"
  implies the positive and negative correlations, "abs" takes the absolute value of
  both to combine together and the "blended" = original image * 0.2 + abs * 0.8.
