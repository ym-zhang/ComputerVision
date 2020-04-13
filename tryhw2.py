from uwimg import *
im1 = load_image("data/Rainier1.png")
im2 = load_image("data/Rainier2.png")
pan = panorama_image(im1, im2, thresh=50)
save_image(pan, "easy_panorama")
