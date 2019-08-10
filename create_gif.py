import os, PIL, imageio, glob

# def display_image(epoch_no):
#     return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

EPOCHS = 50
anim_file = os.path.join("gif", 'dcgan.gif')
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(os.path.join("generated_mnist", 'image*.png'))
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)




