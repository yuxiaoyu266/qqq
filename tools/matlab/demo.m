%·Öpatch
clc;
clear;
img_path = '..\datasets\SIQAD\DistortedImages\';
index_path = '..\datasets\index\';
patch_root = '..\datasets\patch\';
img_dir = dir([img_path '*.bmp']);
for i = 1:length(img_dir)
    img = imread([img_path img_dir(i).name]);
    ref = double(rgb2gray(img));
    [I_r,I_c] = size(ref);
    save_name = strsplit(img_dir(i).name,'.');
    name = strsplit(img_dir(i).name,'_');
    index_name = [index_path name{1} '.bmp'];
    index = double(imread(index_name));
    center = kmeans_text(index, 6);
    [r,c] = size(center);
    for j = 1:r
        y = int32(center(j,1));
        x = int32(center(j,2));
        c_l = x-112;
        c_r = x+112;
        r_t = y-112;
        r_b = y+112;
        if c_l < 1
            c_r = c_r - c_l + 1;
            c_l = 1;
        end
        if c_r > I_c
            c_l = c_l - (c_r - I_c);
            c_r = I_c;
        end
        if r_t < 1
            r_b = r_b - r_t + 1;
            r_t = 1;
        end
        if r_b > I_r
            r_t = r_t - (r_b - I_r);
            r_b = I_r;
        end
        
        save_path = [patch_root save_name{1} '\text\'];
        if j > r/2
            save_path = [patch_root save_name{1} '\pic\'];
        end
        if exist(save_path)==0 
            mkdir(save_path);
        end
        im = img(r_t:r_b,c_l:c_r,1:3);     
        imwrite(im,[save_path num2str(j) '.bmp'])    
    end
    fprintf('%s.bmp is finish\n',save_name{1});
end