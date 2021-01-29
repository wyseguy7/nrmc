import os
import pickle
import multiprocessing as mp
import functools
import argparse
import re


from networkx.drawing.nx_pylab import draw
from matplotlib import pyplot as plt
import pandas as pd
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--iter_per_image', action='store', type=int, required=False, default=1000)
parser.add_argument('--skip_images', action='store_true', default=False)


args = parser.parse_args()
overwrite = args.overwrite == 'yes'

def write_movie(filepath, iter_per_image=1000, overwrite=False, skip_images=False):

    folder_path = os.path.split(filepath)[0]
    img_path = os.path.join(folder_path, 'img')
    os.makedirs(img_path, exist_ok=True) # guarantee this exists
    out_path = os.path.join(folder_path, 'movie.avi')

    if os.path.exists(out_path) and not overwrite:
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    counter = 0

    # make the images based on the state
    if not skip_images:
        for move in process.state.move_log:
            if counter % iter_per_image == 0:
                f = plt.figure(figsize=(8,8)) # should this be editable?

                draw(process._initial_state.graph,
                     pos={node_id: (process._initial_state.graph.nodes()[node_id]['Centroid'][0],
                                    process._initial_state.graph.nodes()[node_id]['Centroid'][1]) for node_id in process._initial_state.graph.nodes()},
                     node_color=[process._initial_state.node_to_color[i] for i in process._initial_state.graph.nodes()],
                     node_size=100)

                filepath = os.path.join(img_path, '{}.png'.format(counter))
                f.savefig(filepath)
                plt.close()

            if move is not None:
                process._initial_state.handle_move(move)

            counter += 1

    # now make the movie out of the images

    # guarantee that we sort correctly according to number
    files = sorted(os.listdir(img_path), key=lambda x: int(re.sub('\\..*','', x)))
    print(files)
    img = [cv2.imread(os.path.join(img_path, i)) for i in os.listdir(img_path) if '.png' in i] # the list of images
    height, width, layers = img[0].shape
    video = cv2.VideoWriter(out_path, -1, 1, (width, height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows() # this this going to lead to bad memory issues if not run periodically?
    video.release()



func = functools.partial(write_movie, overwrite=overwrite, iter_per_image=args.iter_per_image)
df = pd.read_csv(args.filepaths)


def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
