clc
clear
close all
%% Path setting
% Check the image format in the folder is correct (png, jpg...)!
% Denoise image path (end with '\'):
%pathD = 'C:\Users\Lab722 BX\Desktop\TEST\deblur\';
pathD = 'D:\PycharmProjects\SRMNet-main\test_results\McMaster\50\';
img_denoise_list = dir(strcat(pathD,'*.png'));

% Noise image path (end with '\'):  
%pathN = '\';
pathN = 'D:\PycharmProjects\SRMNet-main\test_dataset\McMaster\50\';
img_noise_list = dir(strcat(pathN,'*.png'));

% Clean image path (end with '\'): 
%pathGT = 'D:\NCHU\paper submit\ISCAS 2022\Other methods\CBSD68_test\CBSD68_GT_same_size\';
pathGT = 'D:\NCHU\Dataset\Denoise\McMaster\';
img_GT_list = dir(strcat(pathGT,'*.tif')); 
%% Evaluation
noise_psnr = 0;
denoise_psnr = 0;
noise_ssim = 0;
denoise_ssim = 0;

img_num = length(img_GT_list);%獲取影象總數量


for j = 1:img_num %逐一讀取影象
	GT_name = img_GT_list(j).name;% 影象名
    GT = imread(strcat(pathGT,GT_name));
    %GT = im2single(GT);
    %GT_g = rgb2gray(GT);
        
    Noise_name = img_noise_list(j).name;% 影象名
    Noise = imread(strcat(pathN,Noise_name));
    %Noise = im2single(Noise);
    %Noise_g = rgb2gray(Noise);
        
    Denoise_name = img_denoise_list(j).name;% 影象名
    Denoise = imread(strcat(pathD,Denoise_name));
    %Denoise = im2single(Denoise);
    %Denoise_g = rgb2gray(Denoise);
    
    d_psnr = psnr(Denoise, GT); 
    n_psnr = psnr(Noise, GT); 
    d_ssim = ssim(Denoise, GT);
    n_ssim = ssim(Noise, GT);
    
    fprintf('GT: %s\n' ,GT_name);
    fprintf('Noise: %s\n' ,Noise_name);
    fprintf('Denoise: %s\n' ,Denoise_name);
    fprintf('\n')
    fprintf('  Noise PSNR  = %0.4f dB ---------- %2d/%2d\n', n_psnr, j, img_num);
    fprintf('  Noise SSIM  = %0.4f     ---------- %2d/%2d\n', n_ssim, j, img_num);
    fprintf('Denoise PSNR* = %0.4f dB ---------- %2d/%2d\n', d_psnr, j, img_num);
    fprintf('Denoise SSIM* = %0.4f     ---------- %2d/%2d\n', d_ssim, j, img_num);
    fprintf('----------------------------------------------\n')
    noise_psnr = noise_psnr + n_psnr;
    denoise_psnr = denoise_psnr + d_psnr;
    noise_ssim = noise_ssim + n_ssim;
    denoise_ssim = denoise_ssim + d_ssim;
end       
Total_noise_PSNR = noise_psnr/img_num;
Total_denoise_PSNR = denoise_psnr/img_num;
Total_noise_SSIM = noise_ssim/img_num;
Total_denoise_SSIM = denoise_ssim/img_num;
fprintf('                     Finsh!                   \n');
fprintf('  Average noise PSNR  = %0.4f dB\n', Total_noise_PSNR);
fprintf('  Average noise SSIM  = %0.4f\n', Total_noise_SSIM);
fprintf('Average denoise PSNR* = %0.4f dB\n', Total_denoise_PSNR);
fprintf('Average denoise SSIM* = %0.4f\n', Total_denoise_SSIM);
fprintf('----------------------------------------------\n')
