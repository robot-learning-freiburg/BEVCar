"""
code for image precomputation -> scales the full res images down to the specified scaling factor
"""

import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image


def grab_convert_store_imgs(version, dataroot, is_train, target_dir, new_w, new_h):
    # load nuscenes
    nusc = NuScenes(version='v1.0-{}'.format(version), dataroot=dataroot, verbose=True)
    print("... data loaded! \n")

    # filter by scene split
    split = {
        'v1.0-trainval': {True: 'train', False: 'val'},
        'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
    }[nusc.version][is_train]
    # get list of scene strings for specified split
    scenes = create_splits_scenes()[split]

    # get indices for relevant samples -> based on chosen split
    samples = [samp for samp in nusc.sample]
    # remove samples that aren't in this split
    samples = [samp for samp in samples if nusc.get('scene', samp['scene_token'])['name'] in scenes]
    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # w = 1600
    # h = 900

    final_dim = (new_w, new_h)

    res_str = str(final_dim[0]) + "_" + str(final_dim[1])
    new_target_dir = os.path.join(target_dir, res_str)
    if not os.path.exists(new_target_dir):
        os.mkdir(new_target_dir)

    samples_dir = os.path.join(new_target_dir, "samples")
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)

    sample_num = 0
    for sample in samples:
        print("Sample: %d" % sample_num)

        for cam in cams:
            samp = nusc.get('sample_data', sample['data'][cam])
            cam_name = cam
            cam_dir = os.path.join(samples_dir, cam_name)
            if not os.path.exists(cam_dir):
                os.mkdir(cam_dir)
            imgname = os.path.join(dataroot, samp['filename'])
            new_imgname = os.path.join(new_target_dir, samp['filename'])

            img = Image.open(imgname)
            img_scaled = img.resize(size=final_dim, resample=Image.NEAREST)

            # save scaled image:
            img_scaled.save(new_imgname)

        sample_num += 1
    return


if __name__ == '__main__':
    version = "trainval"  # specifies the split that is loaded
    print("Version for conversion: '%s' " % version)
    dataroot = "../../../nuscenes/nuscenes"  # path to NuScenes directory
    print("Data is taken from '%s'" % dataroot)
    target_dir = "../../../nuscenes/scaled_images"   # custom data path
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print("Directory '%s' created" % target_dir)
    else:
        print("Directory '%s' already exists!" % target_dir)

    # get the image -> convert to the desired resolution -> store data at desired path
    grab_convert_store_imgs(version=version,
                            dataroot=dataroot,
                            is_train=True,  # True -> Trainset ; False -> Valset
                            target_dir=target_dir,
                            new_w=448, new_h=896)  # change desired image resolution
