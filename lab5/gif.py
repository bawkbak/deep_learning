import glob
from PIL import Image

fp_in = "./result_seq/7/compare_*.png"
fp_out = "./result_seq/7/result_compare.gif"

imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)