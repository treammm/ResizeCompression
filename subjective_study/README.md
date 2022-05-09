# Subjective study

This folder provides the results of human subject study in Section IV-E of our paper.

## Download the (cropped) reference and compressed images
```
wget -O study_images_crop_all.zip https://utexas.box.com/shared/static/izd3wopcohce7htgox0cb55v5netni7l.zip
unzip study_images_crop_all.zip
rm -r study_images_crop_all.zip __MACOSX
```

## Subjective scores
Columns 8 and 9 in `study_results.csv` stores the subjective scores obtained from 47 subjects.

## Objective scores
Columns 10--15 in `study_results.csv` stores the objective scores (PSNR_Y, VMAF0.6.1, and, VIF_pixel) obtained from image pairs.
