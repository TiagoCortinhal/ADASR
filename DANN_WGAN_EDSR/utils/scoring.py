import os

import cv2
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from utils.fid_score import calculate_fid_given_paths as fid
from utils.options import args


def scoring():
    hr = args.output_dir + '/hr' + str(args.factor) + "/"
    lr = args.output_dir + '/lr' + str(args.factor) + "/"
    eval = args.output_dir + '/' + str(args.eval_epoch) + "_" + str(args.factor) + '/eval/'
    reals = os.listdir(hr)
    path = args.output_dir + '/' + str(args.eval_epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = open(
        args.output_dir + '/' + str(args.eval_epoch) + "_" + str(args.factor) + '/' + str(args.eval_epoch) + "_" + str(
            args.factor) + '.txt', 'a+')
    generated = os.listdir(eval)
    reals = sorted(reals)
    generated = sorted(generated)
    psnr_sum = 0
    ssim_sum = 0

    for r, g in zip(reals, generated):
        name = g
        r = cv2.imread(hr + r)
        g = cv2.imread(eval + g)
        g = cv2.resize(g, (r.shape[1], r.shape[0]))

        ##TODO fix this
        cv2.imwrite(args.output_dir + '/' + str(args.eval_epoch) + "_" + str(args.factor) + '/resized_' + name, g)

        psnr_ = psnr(r, g)
        ssim_ = ssim(r, g, multichannel=True, sigma=1.5, use_sample_covariance=False, gaussian_weights=True)
        psnr_sum += psnr_
        ssim_sum += ssim_
        log.write('PNSR:{}\tSSIM:{}\n'.format(psnr_, ssim_))
    fid_score = fid((hr, eval), dims=2048, cuda='', batch_size=1)
    log.write("\nFID:\t{}\n".format(fid_score))
    log.write('AVG_PNSR:{}\tAVG_SSIM:{}\n'.format(psnr_sum / len(generated), ssim_sum / len(generated)))
