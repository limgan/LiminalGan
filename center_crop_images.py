# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:29:54 2020

@author: limgan

"""






import hashlib
#from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np

from PIL import Image
import os
import numpy as np

def crop_image(im):
    w = im.width
    h = im.height
    change = ''
    if w > h:
        new_width = h
        new_height = h
        change = 'w'
    if w < h:
        new_height = w
        new_width = w
        change = 'h'
        
    if w==h:
        return im
        
    left = int((w - new_width)/2)
    top = int((h - new_height)/2)
    right = int((w + new_width)/2)
    bottom = int((h + new_height)/2)
    
    if bottom - top == right-left:
        print('square')
        print((right-left, bottom - top ))
        return im.crop((left, top, right, bottom))
    else:
        print('unsquare')
        print((right-left, bottom - top ))
        diff = (bottom - top) - (right-left)
        if change == 'w':
            right = right - diff
        if change == 'h':
            bottom = bottom - diff
        return im.crop((left, top, right, bottom))

# os.chdir('D:/datasets/102flowers/')
# target_dir = 'D:/datasets/cropped_flowers/'
# m = 0
# for file in os.listdir("D:/datasets/102flowers/"):
#     im = Image.open(file)
#     cropped_im = crop_image(im)
#     fname = os.path.splitext(os.path.basename(file))[0]
#     cropped_im.save( (target_dir + str(m) + '_c.png' ), 'PNG'  )
#     m+=1


def resize_to_2(im):
    powers = np.array([2**x for x in range(0,14)])
    w,h = im.size
    newsize = powers[np.argmin(np.abs(powers - w))]
    return im.resize((newsize,newsize), Image.ANTIALIAS)
    
    
# os.chdir('D:/datasets/cropped_flowers/')
# target_dir = 'D:/datasets/cropped_resized_flowers/'

# m = 0 
# for file in os.listdir("."):
#     im = Image.open(file)
#     resized_im = resize_to_2(im)
#     fname = os.path.splitext(os.path.basename(file))[0]
#     print(resized_im.size)
#     resized_im.save( (target_dir + str(m) + '_c_r.png' ), 'PNG'  )
#     m+=1
    
    

# target_dir = 'D:/datasets/flowers128/'
# os.chdir('D:/datasets/cropped_resized_flowers/')
# lens = []
# m = 0
# for file in os.listdir('.'):
#     im = Image.open(file)
#     if im.size[0] >= 128:
#         if im.size[0] != 128:
#             im = im.resize((128,128), Image.ANTIALIAS)
#         im.save( (target_dir + str(m) + '_c_r.png' ), 'PNG'  )
#     m+=1


# os.chdir('D:/liminal_spaces/L512/')
# lens = []
# m = 0
# for file in os.listdir('.'):
#     im = Image.open(file) 
#     if im.mode !='RGB':
#         im = im.convert('RGB')
#         im.save( file, 'PNG'  )


             

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im



    
def check_ims(d):
    for file in os.listdir("D:/liminal_spaces/reddit_sub_LiminalSpace/liminal_spaces_cropped"):
        im = Image.open(file)
        print(im.size)
        
        
def rename_to_numbers():
    os.chdir('D:/liminal_spaces/cropped_images/')
    FList = os.listdir('D:/liminal_spaces/cropped_images/')
    FListC = FList[1:]
    
    m = 0
    for i in range(0,len(FListC)):
        fileExtension = os.path.splitext(FListC[i])[1]
        os.rename(FListC[i],str(m)+fileExtension)
        m = m+1
        
        
        
        
        
        
        
        
        
def make_dataset(in_dir, out_dir, resolution_limit):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    m = 0
    print('processing %d files....' % len(os.listdir(in_dir) )) 
    for file in os.listdir(in_dir):
        im = Image.open(  (in_dir + file)   )
        
        new_image = remove_transparency(resize_to_2(crop_image(im)))
        
      
        if new_image.size[0] >= resolution_limit:
            if new_image.size[0] != resolution_limit:
                new_image = new_image.resize((resolution_limit,resolution_limit), Image.ANTIALIAS)
        
            fname = os.path.splitext(os.path.basename(file))[0]
            
            if new_image.mode !='RGB':
                new_image = new_image.convert('RGB')
                
            new_image.save( (out_dir + str(m) + '_processed.png' ), 'PNG'  )
            m+=1
        

    dirs = [out_dir]
    
    os.getcwd()
    
    file_list = []
    
    for d in dirs:
        file_list.append(  [(d  + x) for x in  os.listdir(d)]  )
    
    flat_list = [item for sublist in file_list for item in sublist]
    
    duplicates = []
    hash_keys = dict()
    for index, filename in  enumerate(flat_list):  #listdir('.') = current directory
    
    
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys: 
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))
                
    print('removing %d duplicates....'%len(duplicates))
    for index in duplicates:
        for i in range(len(index)-1):
            os.remove(flat_list[index[i]])
            
    print('Total files in final dataset: %d '%len(flat_list))

    return 'Done'

        
        
        
        
make_dataset('./liminal_gan_master_photos/master_outdoors/', './L512_new_outdoors/', 512 )
        
        
        
        
        
        
        
        