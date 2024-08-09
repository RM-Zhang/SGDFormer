import os
import cv2
import torch
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from SGDFormer import SGDFormer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False


def add_noise(src, alpha, sigma):
    if not alpha == 0:
        src = alpha * np.random.poisson(src / alpha).astype(float)
    noise = np.random.normal(0, sigma, src.shape)
    src = src + noise
    src = np.clip(src, 0, 1.0)
    return src


def main():
    # Set Noise Level
    alpha = 0.02
    sigma = 0.2

    # Load model
    local_range = 5 # the window size of neighboorhood attention in the NRCA module
    trans_num = 1   # the number of transformer block
    model = SGDFormer(img_channel = 1, width = 32, max_disp = 128, local_range = local_range, trans_num = trans_num).cuda()
    model.load_state_dict(torch.load('ckpt/SGDFormer-PittsStereo.pth'))
    model.eval()
    lpfunc = lpips.LPIPS(net='vgg').cuda()

    # Load Image - PittsStereo RGB-NIR
    target_img = cv2.imread('examples/PittsStereo/001_target.png').astype('float') / 255.0
    guidance_img = cv2.imread('examples/PittsStereo/001_guidance.png').astype('float') / 255.0

    # Add Noise
    noisy_img = add_noise(target_img, alpha, sigma)
    noisy_img = np.clip(noisy_img, 0, 1.0)

    # Convert Images Into Tensors
    target_img = torch.from_numpy(np.ascontiguousarray(target_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()
    guidance_img = torch.from_numpy(np.ascontiguousarray(guidance_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()
    noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float().unsqueeze(0).cuda()

    # Start Denoising
    with torch.no_grad():
        denoised_img = model(noisy_img.permute(1,0,2,3), guidance_img.permute(1,0,2,3)).permute(1,0,2,3)
    denoised_img = torch.clamp(denoised_img, 0, 1.0)

    # Compute PSNR, SSIM, & LPIPS
    lpips_value = lpfunc(denoised_img, target_img).item()

    denoised_img = denoised_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    target_img = target_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    noisy_img = noisy_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    psnr_value = psnr(denoised_img, target_img)
    ssim_value = ssim(denoised_img, target_img, channel_axis=2)

    print('==========================================================================')
    print('PSNR={}, SSIM={}, LPIPS={}'.format(psnr_value, ssim_value, lpips_value))
    print('==========================================================================')

    # Save Images
    cv2.imwrite('results/001_PittsStereo_res.png', np.uint8(denoised_img * 255))
    cv2.imwrite('results/001_PittsStereo_noisy.png', np.uint8(noisy_img * 255))

if __name__ == "__main__":
    main()