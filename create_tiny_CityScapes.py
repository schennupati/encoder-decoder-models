import os
import argparse

from PIL import Image
from tqdm import tqdm


def save_resized_file(output_folder, file_path, imsize=[256, 128]):
    if file_path.split('.')[-1] in ['json', 'npz']:
        return 

    img = Image.open(file_path)
    resized_image = img.resize(imsize, resample=Image.NEAREST)
    new_path = os.path.join(output_folder,
                            file_path.split('/')[-1])
    resized_image.save(new_path)


def main(root_folder, output_folder):
    # get directories of the cities
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        return 
    temp_files = os.listdir(root_folder)

    for file_ in tqdm(temp_files):
        save_resized_file(output_folder, os.path.join(root_folder, file_))

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('root_folder')
    #parser.add_argument('output_folder')
    #args = vars(parser.parse_args())
    root_path = '/home/sumche/datasets/Cityscapes' #args['root_folder']
    out_root_path = '/home/sumche/datasets/Tiny_Cityscapes' #args['output_folder']
    folders = ['leftImg8bit', 'gtFine']
    for folder in folders:
        path = os.path.join(root_path, folder)
        out_path = os.path.join(out_root_path, folder)  
        for root, dirs, names in os.walk(path, topdown=False):
            for dir in dirs:
                split = root.split('/')[-1]
                if split in ['train', 'val']:
                    main(os.path.join(root,dir), os.path.join(out_path, split, dir))