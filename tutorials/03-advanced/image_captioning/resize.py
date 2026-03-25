import argparse
import os
from PIL import Image


def resize_image(image, size):
    """将图像调整为给定大小。"""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """调整 'image_dir' 中图像的大小并保存到 'output_dir' 中。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] 已调整图像大小并保存到 '{}' 中。"
                   .format(i+1, num_images, output_dir))

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='训练图像目录')
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='保存调整大小后的图像的目录')
    parser.add_argument('--image_size', type=int, default=256,
                        help='处理后图像的大小')
    args = parser.parse_args()
    main(args)
