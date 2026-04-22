import os
import os.path as op
# from typing import List

# from utils.iotools import read_json
from .bases import BaseDataset

class VCM(BaseDataset):
    dataset_dir = 'vcm'
    """
    VCM dataset for IRRA baseline with minimal changes.

    Directory example:
        root/
            Train/
                0001/
                    rgb/
                        D1/
                            caption.txt
                            mosaic_for_caption.jpg
                            frame_xxx.jpg
                            ...
                        D2/
                        D3/
                    ir/
                        D1/
                        D2/
                        D3/
            Test/
                0503/
                ...
    """

    def __init__(self, root='', verbose=True):
        super().__init__()
        self._debug_print_done = False

        self.dataset_dir = op.join(root, self.dataset_dir)
        self.train_dir = op.join(self.dataset_dir, 'Train')
        self.test_dir = op.join(self.dataset_dir, 'Test')

        self.train_tracklets = []
        self.query_texts = []
        self.gallery_rgb_tracklets = []
        self.gallery_ir_tracklets = []

        self.gallery_rgb_single = []
        self.gallery_ir_single = []
        self.gallery_mixed_single = []
        self.query_single = []

        self.train = []
        self.test = {}
        self.val = {}

        self.train_id_container = set()
        self.test_id_container = set()
        self.val_id_container = set()

        self._check_before_run()

        self._process_split('Train')
        self._process_split('Test')
        self._build_train_single_frame()
        self._build_gallery_single_frame()
        self._build_query_single()

        if verbose:
            print("VCM loaded")
            print("train_tracklets:", len(self.train_tracklets))
            print("query_texts:", len(self.query_texts))
            print("gallery_rgb_tracklets:", len(self.gallery_rgb_tracklets))
            print("gallery_ir_tracklets:", len(self.gallery_ir_tracklets))    
            print("train_id_container:", len(self.train_id_container)) 
            print("test_id_container:", len(self.test_id_container))
            print("query_single:", len(self.query_single))
            print(self.query_single[0])



    def _check_before_run(self):
        required_dirs = [self.dataset_dir, self.train_dir, self.test_dir]

        for path in required_dirs:
            if not op.exists(path):
                raise RuntimeError(f"'{path}' is not available.")
            
    def _collect_frame_paths(self, cam_dir):
        frame_paths = []
        file_names = os.listdir(cam_dir)
        for name in file_names:
            if name == 'caption.txt' or name == 'caption_aug.txt':
                continue
            if name == 'mosaic_for_caption.jpg':
                continue
            lower_name = name.lower()
            if not lower_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = op.join(cam_dir, name)
            if op.isfile(img_path):
                frame_paths.append(img_path)
        frame_paths = sorted(frame_paths)
        return frame_paths
    
    def _sample_middle_frame(self, frame_paths):
        if len(frame_paths) == 0:
            return None
        
        mid_idx = len(frame_paths) // 2
        return frame_paths[mid_idx]
    
    def _build_train_single_frame(self):
        self.train = []
        image_id = 0
        for tracklet in self.train_tracklets:
            pid = tracklet['pid']
            frame_paths = tracklet['all_frame_paths']
            caption = tracklet['caption']

            img_path = self._sample_middle_frame(frame_paths)
            if img_path is None:
                continue

            self.train.append((pid, image_id, img_path, caption))
            image_id += 1

    def _build_gallery_single_frame(self):
        self.gallery_rgb_single = []
        self.gallery_ir_single = []
        self.gallery_mixed_single = []

        for tracklet in self.gallery_rgb_tracklets:
            pid = tracklet['pid']
            camid = tracklet['camid']
            frame_paths = tracklet['all_frame_paths']

            img_path = self._sample_middle_frame(frame_paths)
            if img_path is None:
                continue

            record = {
                'pid': pid,
                'modality': 'rgb',
                'camid': camid,
                'img_path': img_path
            }
            self.gallery_rgb_single.append(record)
        
        for tracklet in self.gallery_ir_tracklets:
            pid = tracklet['pid']
            camid = tracklet['camid']
            frame_paths = tracklet['all_frame_paths']

            img_path = self._sample_middle_frame(frame_paths)
            if img_path is None:
                continue

            record = {
                'pid': pid,
                'modality': 'ir',
                'camid': camid,
                'img_path': img_path
            }
            self.gallery_ir_single.append(record)

            self.gallery_mixed_single = self.gallery_rgb_single + self.gallery_ir_single

    def _build_query_single(self):
        self.query_single = []
        
        for item in self.query_texts:
            pid = item['pid']
            caption = item['caption']

            record = {
                'pid': pid,
                'caption': caption
            }
            self.query_single.append(record)

    def _read_caption(self, cam_dir):
        caption_path = op.join(cam_dir, 'caption.txt')

        if not op.isfile(caption_path):
            return ''

        with open(caption_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        return text
    
    def _process_split(self, split_name):
        if split_name == 'Train':
            split_dir = self.train_dir
        elif split_name == 'Test':
            split_dir = self.test_dir
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        
        pid_list = []
        for name in os.listdir(split_dir):
            pid_path = op.join(split_dir, name)
            if op.isdir(pid_path):
                pid_list.append(name)

        pid_list = sorted(pid_list)

        rgb_pid_set = set()
        ir_pid_set = set()
        rgb_camera_count = 0
        ir_camera_count = 0
        rgb_valid_tracklet_count = 0
        ir_valid_tracklet_count = 0

        for pid in pid_list:
            pid_path = op.join(split_dir, pid)
            rgb_path = op.join(pid_path, 'rgb')
            ir_path = op.join(pid_path, 'ir')
            
            if op.isdir(rgb_path):
                rgb_pid_set.add(pid)

                rgb_camera_list = []
                for cam_name in os.listdir(rgb_path):
                    cam_path = op.join(rgb_path, cam_name)
                    if op.isdir(cam_path):
                        rgb_camera_list.append(cam_name)

                rgb_camera_list = sorted(rgb_camera_list)
                for cam_name in rgb_camera_list:
                    cam_path = op.join(rgb_path, cam_name)
                    frame_paths = self._collect_frame_paths(cam_path)
                    if len(frame_paths) > 0:
                        caption = self._read_caption(cam_path)
                        rgb_valid_tracklet_count += 1

                        if split_name == 'Train':
                            record = {
                                'pid': pid,
                                'modality': 'rgb',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths,
                                'caption': caption
                            }
                            self.train_tracklets.append(record)
                            self.train_id_container.add(pid)

                        elif split_name == 'Test':
                            record = {
                                'pid': pid,
                                'modality': 'rgb',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths
                            }
                            self.gallery_rgb_tracklets.append(record)
                            self.test_id_container.add(pid)
                            caption = self._read_caption(cam_path)
                            query_record = {
                                'pid': pid,
                                'caption': caption
                            }
                            self.query_texts.append(query_record)
                rgb_camera_count += len(rgb_camera_list)

            if op.isdir(ir_path):
                ir_pid_set.add(pid)

                ir_camera_list = []
                for cam_name in os.listdir(ir_path):
                    cam_path = op.join(ir_path, cam_name)
                    if op.isdir(cam_path):
                        ir_camera_list.append(cam_name)

                ir_camera_list = sorted(ir_camera_list)
                for cam_name in ir_camera_list:
                    cam_path = op.join(ir_path, cam_name)
                    frame_paths = self._collect_frame_paths(cam_path)
                    if len(frame_paths) > 0:
                        caption = self._read_caption(cam_path)
                        ir_valid_tracklet_count += 1

                        if split_name == 'Train':
                            record = {
                                'pid': pid,
                                'modality': 'ir',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths,
                                'caption': caption
                            }
                            self.train_tracklets.append(record)
                            self.train_id_container.add(pid)
                        
                        elif split_name == 'Test':
                            record = {
                                'pid': pid,
                                'modality': 'ir',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths
                            }
                            self.gallery_ir_tracklets.append(record)
                            self.test_id_container.add(pid)
                ir_camera_count += len(ir_camera_list)

        print(f"{split_name} pid count: {len(pid_list)}")
        print(f"{split_name} rgb pid count: {len(rgb_pid_set)}")
        print(f"{split_name} ir pid count: {len(ir_pid_set)}")
        print(f"{split_name} rgb camera count: {rgb_camera_count}")
        print(f"{split_name} ir camera count: {ir_camera_count}")
        print(f"{split_name} rgb valid tracklet count: {rgb_valid_tracklet_count}")
        print(f"{split_name} ir valid tracklet count: {ir_valid_tracklet_count}")
        
        
