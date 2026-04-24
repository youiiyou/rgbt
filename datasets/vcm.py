import os
import os.path as op
import numpy as np
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

    def __init__(self, root='', verbose=True, num_frames=1, train_caption_mode='double'):
        super().__init__()
        self._debug_print_done = False

        if train_caption_mode not in ('single', 'double'):
            raise ValueError(f"Invalid train_caption_mode: {train_caption_mode}")

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

        self.train_multi = []
        self.gallery_rgb_multi = []
        self.gallery_ir_multi = []
        self.gallery_mixed_multi = []

        self.train = []
        self.test = {}
        self.val = {}

        self.train_id_container = set()
        self.test_id_container = set()
        self.val_id_container = set()

        self.train_pid2label = {}
        self.test_pid2label = {}

        self.num_frames = num_frames
        self.train_caption_mode = train_caption_mode

        self._check_before_run()

        self._process_split('Train')
        self._process_split('Test')
        self._build_train_single_frame()
        self._build_gallery_single_frame()
        self._build_query_single()
        self._build_train_multi_frame(self.num_frames)
        self._build_gallery_multi_frame(self.num_frames)

        if self.num_frames == 1:
            self._build_eval_single_frame_dict()
        else:
            self.train = self.train_multi
            self._build_eval_multi_frame_dict()

        if verbose:
            caption_stats = self._compute_train_caption_stats()
            print("VCM loaded")
            print("train:", len(self.train))
            print("query_single:", len(self.query_single))
            print("gallery_mixed_single:", len(self.gallery_mixed_single))
            print("num_train_ids:", len(self.train_id_container))
            print("num_test_ids:", len(self.test_id_container))
            print("train_caption_mode:", self.train_caption_mode)
            print("avg rgb captions per train tracklet:", f"{caption_stats['avg_rgb']:.4f}")
            print("avg ir captions per train tracklet:", f"{caption_stats['avg_ir']:.4f}")
            print("max rgb captions:", caption_stats['max_rgb'])
            print("max ir captions:", caption_stats['max_ir'])
            print("len(train_tracklets):", len(self.train_tracklets))
            print("len(train_multi):", len(self.train_multi))
            print("len(query_single):", len(self.query_single))
 

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
        frame_paths = sorted(
            frame_paths,
            key=lambda x: int(op.splitext(op.basename(x))[0])
        )
        return frame_paths
    
    def _sample_middle_frame(self, frame_paths):
        if len(frame_paths) == 0:
            return None
        
        mid_idx = len(frame_paths) // 2
        return frame_paths[mid_idx]
    
    def _sample_fixed_frames(self, frame_paths, num_frames):
        if num_frames == 0:
            return []
        elif num_frames == 1:
            mid_path = self._sample_middle_frame(frame_paths)
            return [mid_path] if mid_path is not None else []
        
        num_total = len(frame_paths)

        if num_total >= num_frames:
            sampled = []
            for i in range(num_frames):
                start = int(i * num_total / num_frames)
                end = int((i + 1) * num_total / num_frames)
                if end <= start:
                    end = start + 1
                mid = (start + end - 1) // 2
                sampled.append(frame_paths[mid])
            return sampled
        else:
            indices = np.linspace(0, num_total - 1, num_frames)
            indices = [int(round(x)) for x in indices]
            sampled = [frame_paths[idx] for idx in indices]
            return sampled
    
    def _build_train_single_frame(self):
        self.train = []
        image_id = 0
        for tracklet in self.train_tracklets:
            pid = tracklet['pid']
            frame_paths = tracklet['all_frame_paths']
            captions = tracklet['captions']

            img_path = self._sample_middle_frame(frame_paths)
            if img_path is None:
                continue

            for caption in captions:
                self.train.append((pid, image_id, img_path, caption))

            image_id += 1

    def _build_train_multi_frame(self, num_frames):
        self.train_multi = []
        image_id = 0

        for tracklet in self.train_tracklets:
            pid = tracklet['pid']
            frame_paths = tracklet['all_frame_paths']
            captions = tracklet['captions']

            sampled_paths = self._sample_fixed_frames(frame_paths, num_frames)
            if len(sampled_paths) != num_frames:
                continue

            for caption in captions:
                self.train_multi.append((pid, image_id, sampled_paths, caption))
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

    def _build_gallery_multi_frame(self, num_frames):
        self.gallery_rgb_multi = []
        self.gallery_ir_multi = []
        self.gallery_mixed_multi = []

        for tracklet in self.gallery_rgb_tracklets:
            pid = tracklet['pid']
            camid = tracklet['camid']
            frame_paths = tracklet['all_frame_paths']

            sampled_paths = self._sample_fixed_frames(frame_paths, num_frames)
            if len(sampled_paths) != num_frames:
                continue

            record = {
                'pid': pid,
                'modality': 'rgb',
                'camid': camid,
                'img_paths': sampled_paths
            }
            self.gallery_rgb_multi.append(record)

        for tracklet in self.gallery_ir_tracklets:
            pid = tracklet['pid']
            camid = tracklet['camid']
            frame_paths = tracklet['all_frame_paths']

            sampled_paths = self._sample_fixed_frames(frame_paths, num_frames)
            if len(sampled_paths) != num_frames:
                continue

            record = {
                'pid': pid,
                'modality': 'ir',
                'camid': camid,
                'img_paths': sampled_paths
            }
            self.gallery_ir_multi.append(record)

        self.gallery_mixed_multi = self.gallery_rgb_multi + self.gallery_ir_multi

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

    def _build_eval_single_frame_dict(self):
        image_pids = []
        img_paths = []
        caption_pids = []
        captions = []
        for item in self.gallery_mixed_single:
            image_pids.append(item['pid'])
            img_paths.append(item['img_path'])

        for item in self.query_single:
            caption_pids.append(item['pid'])
            captions.append(item['caption'])

        self.test = {
            'image_pids': image_pids,
            'img_paths': img_paths,
            'caption_pids': caption_pids,
            'captions': captions,
        }

        self.val = {
            'image_pids': list(image_pids),
            'img_paths': list(img_paths),
            'caption_pids': list(caption_pids),
            'captions': list(captions),
        }
        self.train_annos = self.train
        self.test_annos = self.gallery_mixed_single
        self.val_annos = self.gallery_mixed_single

        self.val_id_container = set(self.test_id_container)

    def _build_eval_multi_frame_dict(self):
        image_pids = []
        img_paths = []
        caption_pids = []
        captions = []

        for item in self.gallery_mixed_multi:
            image_pids.append(item['pid'])
            img_paths.append(item['img_paths'])   # 这里是 list[str]

        for item in self.query_single:
            caption_pids.append(item['pid'])
            captions.append(item['caption'])

        self.test = {
            'image_pids': image_pids,
            'img_paths': img_paths,
            'caption_pids': caption_pids,
            'captions': captions,
        }

        self.val = {
            'image_pids': list(image_pids),
            'img_paths': list(img_paths),
            'caption_pids': list(caption_pids),
            'captions': list(captions),
        }

        self.train_annos = self.train
        self.test_annos = self.gallery_mixed_multi
        self.val_annos = self.gallery_mixed_multi
        self.val_id_container = set(self.test_id_container)

    def _read_caption_file(self, cam_dir, filename):
        path = op.join(cam_dir, filename)
        if not op.isfile(path):
            return ''

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        return text

    def _read_captions(self, cam_dir, include_aug=True, max_captions=2):
        captions = []
        caption_files = ['caption.txt']
        if include_aug:
            caption_files.append('caption_aug.txt')

        for name in caption_files:
            path = op.join(cam_dir, name)
            if not op.isfile(path):
                continue

            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if text != '':
                captions.append(text)

        # 去重，避免两个文件内容完全一样
        captions = list(dict.fromkeys(captions))
        if max_captions is not None:
            captions = captions[:max_captions]
        return captions

    def _read_train_captions(self, cam_dir):
        if self.train_caption_mode == 'single':
            return self._read_captions(cam_dir, include_aug=False, max_captions=1)

        return self._read_captions(cam_dir, include_aug=True, max_captions=2)

    def _compute_train_caption_stats(self):
        rgb_caption_counts = []
        ir_caption_counts = []

        for tracklet in self.train_tracklets:
            captions = tracklet.get('captions', [])
            count = len(captions)
            modality = tracklet.get('modality')
            if modality == 'rgb':
                rgb_caption_counts.append(count)
            elif modality == 'ir':
                ir_caption_counts.append(count)

        rgb_avg = (sum(rgb_caption_counts) / len(rgb_caption_counts)) if rgb_caption_counts else 0.0
        ir_avg = (sum(ir_caption_counts) / len(ir_caption_counts)) if ir_caption_counts else 0.0

        return {
            'avg_rgb': rgb_avg,
            'avg_ir': ir_avg,
            'max_rgb': max(rgb_caption_counts) if rgb_caption_counts else 0,
            'max_ir': max(ir_caption_counts) if ir_caption_counts else 0,
        }
    
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

        if split_name == 'Train':
            self.train_pid2label = {pid: idx for idx, pid in enumerate(pid_list)}
        elif split_name == 'Test':
            self.test_pid2label = {pid: idx for idx, pid in enumerate(pid_list)}

        rgb_pid_set = set()
        ir_pid_set = set()
        rgb_camera_count = 0
        ir_camera_count = 0
        rgb_valid_tracklet_count = 0
        ir_valid_tracklet_count = 0

        for pid in pid_list:
            if split_name == 'Train':
                pid_label = self.train_pid2label[pid]
            else:
                pid_label = self.test_pid2label[pid]
            pid_path = op.join(split_dir, pid)
            rgb_path = op.join(pid_path, 'rgb')
            ir_path = op.join(pid_path, 'ir')
            
            canonical_rgb_captions = []
            canonical_rgb_selected = False

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
                        if split_name == 'Train':
                            captions = self._read_train_captions(cam_path)
                        else:
                            captions = []

                        if split_name == 'Train' and (not canonical_rgb_selected) and len(captions) > 0:
                            canonical_rgb_captions = list(captions)
                            canonical_rgb_selected = True
                        rgb_valid_tracklet_count += 1

                        if split_name == 'Train':
                            record = {
                                'pid': pid_label,
                                'modality': 'rgb',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths,
                                'captions': captions
                            }
                            self.train_tracklets.append(record)
                            self.train_id_container.add(pid_label)

                        elif split_name == 'Test':
                            record = {
                                'pid': pid_label,
                                'modality': 'rgb',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths
                            }
                            self.gallery_rgb_tracklets.append(record)
                            self.test_id_container.add(pid_label)

                            query_captions = self._read_captions(cam_path, include_aug=False, max_captions=1)
                            if len(query_captions) > 0:
                                query_record = {
                                    'pid': pid_label,
                                    'caption': query_captions[0]
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
                        ir_valid_tracklet_count += 1

                        if split_name == 'Train':
                            captions = list(canonical_rgb_captions)
                            record = {
                                'pid': pid_label,
                                'modality': 'ir',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths,
                                'captions': captions
                            }
                            self.train_tracklets.append(record)
                            self.train_id_container.add(pid_label)
                        
                        elif split_name == 'Test':
                            record = {
                                'pid': pid_label,
                                'modality': 'ir',
                                'camid': cam_name,
                                'all_frame_paths': frame_paths
                            }
                            self.gallery_ir_tracklets.append(record)
                            self.test_id_container.add(pid_label)
                ir_camera_count += len(ir_camera_list)
        
        
