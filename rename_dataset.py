import os
import shutil

# create destination placeholder
dst_path = 'FSL_images/'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

# static FSL alphabet signs
alphabet_letters = [chr(letter) for letter in range(65,91)]
alphabet_letters.remove('J')
alphabet_letters.remove('Z')

# create subfolders
for letter in alphabet_letters:
    subfolder = os.path.join(dst_path, letter)
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

# copy and rename
src_path = 'Collated/'
for letter in alphabet_letters:
    src_folder = os.path.join(src_path, letter)
    dst_folder = os.path.join(dst_path, letter)
    for idx, filename in enumerate(os.listdir(src_folder)):
        if not os.path.exists(dst_folder+'/'+str(idx)+'.jpg'):
            shutil.copy(src_folder+'/'+filename, dst_folder+'/'+str(idx)+'.jpg')
