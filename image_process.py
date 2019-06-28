import os
import sys
from PIL import Image
import numpy as np

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not input_dir or not output_dir:
        print('Select input & output dir')
        return

    files = os.listdir(input_dir)
    files.sort()
    for filename in files:
        print('Processing %s' % filename)
        im = Image.open(os.path.join(input_dir, filename))
        im = np.array(im)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im[i][j].max()-im[i][j].min() > 0xd0 or np.average(im[i][j]) > 0xf0:
                    for k in range(3):
                        im[i][j][k] = (int(im[i-1][j][k])+int(im[i][j-1][k])) // 2
        Image.fromarray(im).save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    main()