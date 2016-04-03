import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Convert kaggle cifar10 png files to bin format')
parser.add_argument('directory', help='Input directory')
parser.add_argument('outfile', help='Output file')
args = parser.parse_args()

outfile = open(args.outfile, mode='bw')
os.chdir(args.directory)
png_files = os.listdir();

print('Sorting...', end='')
png_files.sort(key=lambda x: int(x[:len(x)-4]))
print('done')

print('Writing...')
for i in range(len(png_files)):
    with Image.open(png_files[i]) as im:
        outfile.write(b'\xFF') ## label is not known
        r = im.getdata(0)
        g = im.getdata(1)
        b = im.getdata(2)
        outfile.write(bytes(r))
        outfile.write(bytes(g))
        outfile.write(bytes(b))
    if i % 10000 == 0:
        print(i)
outfile.close()
print('Finished!')
