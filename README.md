# generate_masks
Generate masks for images using [Mask-RCNN](https://github.com/matterport/Mask_RCNN) from matterport. Inputs a folder of images, and outputs a folder containing .npy numpy arrays with the masks for each image.

## Usage
Follow directions from [Mask-RCNN](https://github.com/matterport/Mask_RCNN) to download `mask_rcnn_coco.h5` and pycocotools. I should have a version of pycocotools in the folder, but feel free to replace it if it doesn't work. 

1. Place wanted images in `/images/`
2. Run `python generate_masks.py` to create the masks.
3. To retrieve the masks from `/output_masks/`, use something like `mask = np.load('output_masks/mask_262985539_1709e54576_z.npy')`. 
