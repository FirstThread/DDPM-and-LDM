# Helper script to combine several dataset 

import os
import shutil
import torch
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

def unpack_scene_info_dict(scene_info_dict_fp):
    scene_gt_info = np.load(scene_info_dict_fp, allow_pickle=True).item()[0]
    scene_info = [(val['class_id'], val['px_count_visib']) for val in scene_gt_info.values()]
    scene_info = [item for item in scene_info if item[1] > 0]
    scene_class_ids = [item[0] for item in scene_info]
    return scene_class_ids

ID_DICT = {0: 'background', 1: 'master_chef_can', 2: 'cracker_box', 3: 'sugar_box', 4: 'tomato_soup_can', 5: 'mustard_bottle', 6: 'tuna_fish_can', 7: 'pudding_box', 8: 'gelatin_box', 9: 'potted_meat_can', 10: 'banana', 11: 'strawberry', 12: 'apple', 13: 'lemon', 14: 'peach', 15: 'pear', 16: 'orange', 17: 'plum', 18: 'pitcher_base', 19: 'bleach_cleanser', 20: 'bowl', 21: 'mug', 22: 'sponge', 23: 'plate', 24: 'spatula', 25: 'power_drill', 26: 'wood_block', 27: 'scissors', 28: 'large_marker', 29: 'adjustable_wrench', 30: 'phillips_screwdriver', 31: 'flat_screwdriver', 32: 'hammer', 33: 'medium_clamp', 34: 'large_clamp', 35: 'extra_large_clamp', 36: 'mini_soccer_ball', 37: 'softball', 38: 'baseball', 39: 'tennis_ball', 40: 'racquetball', 41: 'golf_ball', 42: 'foam_brick', 43: 'dice', 44: 'a_cups', 45: 'b_cups', 46: 'c_cups', 47: 'd_cups', 48: 'e_cups', 49: 'f_cups', 50: 'g_cups', 51: 'h_cups', 52: 'i_cups', 53: 'j_cups', 54: 'b_colored_wood_blocks', 55: 'nine_hole_peg_test', 56: 'toy_airplane', 57: 'rubiks_cube', 58: 'bowling_ball', 59: 'folding_table', 60: 'laundry_basket', 61: 'red_bowl', 62: 'shopping_cart', 63: 'wicker_basket', 64: 'wooden_bowl', 65: 'wooden_chair', 66: 'wooden_table', 67: 'metal_chair', 68: 'plastic_chair', 69: 'pool_table', 70: 'art_deco_table', 71: 'wooden_cupboard', 72: 'old_cupboard', 73: 'alex_drawer', 74: 'art_deco_tab', 75: 'kitchen_white_s', 76: 'kitchen_wood_m', 77: 'tall_dresser', 78: 'television_wall_mounted', 79: 'tv_cabinet', 80: 'wooden_drawer', 81: 'bamboo_floor', 82: 'bamboo_wall', 83: 'black_tiling_floor', 84: 'black_tiling_wall', 85: 'black_marble_floor', 86: 'black_marble_wall', 87: 'blackwood_azerocare_marble-4K_floor', 88: 'blackwood_azerocare_marble-4K_wall', 89: 'blue_agathe_marble-4K_floor', 90: 'blue_agathe_marble-4K_wall', 91: 'bluemarble_2-4K_floor', 92: 'bluemarble_2-4K_wall', 93: 'brick_wall_4-4K_floor', 94: 'brick_wall_4-4K_wall', 95: 'brick_wall_11-4K_floor', 96: 'brick_wall_11-4K_wall', 97: 'bronzemarble_1-4K_wall', 98: 'bronzemarble_1_floor', 99: 'cafe_caledonia_granite_floor', 100: 'cafe_caledonia_granite_wall', 101: 'carpet_floor_5_floor', 102: 'carpet_floor_5_wall', 103: 'carpet_floor_6_floor', 104: 'carpet_floor_6_wall', 105: 'carpet_floor_7_floor', 106: 'carpet_floor_7_wall', 107: 'casino_carpet_3_floor', 108: 'casino_carpet_3_wall', 109: 'bedroom', 110: 'kings_room', 111: 'wooden_box', 112: 'suction_gripper', 114: 'red_box'}

if __name__ == '__main__':
    # make sure to never run this file unless you are sure you know what you are doing!
    assert False
    data_root = '/data/BinsceneA_16_objects'
    sub_dirs = [item for item in os.listdir(data_root) if item !='combined']

    # setup everything required to interate over all scenes once
    scene_lists = [sorted(os.listdir(os.path.join(data_root,sub_dir)), key=lambda scene_dir: int(scene_dir)) for sub_dir in sub_dirs]
    scene_lens = [len(scene_list) for scene_list in scene_lists]
    sub_dir_ids = torch.repeat_interleave(torch.arange(len(scene_lens)), torch.tensor(scene_lens))
    scene_ids = torch.cat([torch.arange(scene_len) for scene_len in scene_lens])

    # complete one pass over the dataset to find out which objects were used
    class_ids = np.array([])
    progress = tqdm(range(sum(scene_lens)))
    for i in progress:
        # figure out which scene in which sub-dataset is processed
        sub_dir_idx = sub_dir_ids[i].item()
        scene_idx = scene_ids[i].item()
        # obtain relevant file paths
        scene_path = os.path.join(data_root, sub_dirs[sub_dir_idx], scene_lists[sub_dir_idx][scene_idx], 'SynPick_cam_00_mono')
        scene_gt_info_path = os.path.join(scene_path, 'scene_gt_info.npy')
        # unpack info dict
        class_ids = np.unique(np.append(class_ids, np.array(unpack_scene_info_dict(scene_gt_info_path))))
    print(class_ids)
    print(len(class_ids))

    dataset_squished_classes = -1*np.ones(int(class_ids.max())+1, dtype=np.int64)
    for i, class_id in enumerate(class_ids):
        dataset_squished_classes[int(class_id)] = i
    dataset_labels = [ID_DICT[class_id] for class_id in class_ids.tolist()]
    print(f'Found {len(dataset_labels)} dataset labels:')
    print(dataset_labels)


    # set up file structure for combined dataset
    out_data_dir = os.path.join(data_root, 'combined')
    # recreate the output directory if something is already there
    try:
        shutil.rmtree(out_data_dir)
    except:
        pass
    try:
        os.mkdir(out_data_dir)
    except:
        pass
    print('Writing to:', out_data_dir)
    dataset_info_dir = os.path.join(out_data_dir, 'dataset_info')
    dataset_rgb_dir = os.path.join(out_data_dir, 'rgb')
    dataset_visib_dir = os.path.join(out_data_dir, 'visib')
    os.mkdir(dataset_info_dir)
    os.mkdir(dataset_rgb_dir)
    os.mkdir(dataset_visib_dir)

    # dump labels to dataset_info
    with open(os.path.join(dataset_info_dir, 'classes.json'), "w") as fp:
        json.dump(dataset_labels, fp)

    # once more iterate over the whole dataset, this time create the combined data
    progress = tqdm(range(sum(scene_lens)))
    sample_ids = []
    for i in progress:
        # figure out which scene in which sub-dataset is processed
        sub_dir_idx = sub_dir_ids[i].item()
        scene_idx = scene_ids[i].item()
        # figure out where rgba and visibility vector needs to be saved
        sample_id = f"{i:07d}"
        sample_ids.append(sample_id)
        # obtain relevant file paths
        scene_path = os.path.join(data_root, sub_dirs[sub_dir_idx], scene_lists[sub_dir_idx][scene_idx], 'SynPick_cam_00_mono')
        scene_gt_info_path = os.path.join(scene_path, 'scene_gt_info.npy')
        rgba_img_path = os.path.join(scene_path, 'rgb', '000000.png')
        # unpack info dict
        squished_idx = dataset_squished_classes[torch.tensor(unpack_scene_info_dict(scene_gt_info_path))]
        # re-format to visibility vector and write to file
        visibilty_vec = torch.zeros(len(dataset_labels)).bool()
        visibilty_vec[squished_idx] = 1
        torch.save(visibilty_vec, os.path.join(dataset_visib_dir, sample_id + '.pt'))
        # copy image
        shutil.copyfile(rgba_img_path, os.path.join(dataset_rgb_dir, sample_id + '.png'))
    # finally dump sample ids to dataset_info
    with open(os.path.join(dataset_info_dir, 'sample_ids.json'), "w") as fp:
        json.dump(sample_ids, fp)
