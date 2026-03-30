import numpy as np
import os
import random
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips

random.seed(0)


class DataReaderH36M_CPN(object):
    """
    Human3.6M data reader for CPN detection data (modified version)
    Uses CPN 2D detection results + H36M original 3D annotations, strictly aligned
    """

    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True,
                 dt_root='data/motion3d', dt_file='data_2d_h36m_cpn_ft_h36m_dbb.npz'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_root = dt_root

        # Load CPN 2D data
        cpn_file_path = os.path.join(dt_root, dt_file)
        print(f"Loading CPN 2D data: {cpn_file_path}")
        self.cpn_data = np.load(cpn_file_path, allow_pickle=True)
        self.positions_2d = self.cpn_data['positions_2d'].item()
        self.metadata = self.cpn_data['metadata'].item()

        # Load H36M 3D ground truth data
        h36m_3d_path = os.path.join(dt_root, 'data_3d_h36m.npz')
        print(f"Loading H36M 3D ground truth data: {h36m_3d_path}")
        self.h36m_3d_data = np.load(h36m_3d_path, allow_pickle=True)
        self.positions_3d = self.h36m_3d_data['positions_3d'].item()

        # Define train/test split
        self.train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_subjects = ['S9', 'S11']

        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

        # Camera names (4 views)
        self.camera_names = ['54138969', '60457274', '55011271', '58860488']

        print(f"CPN 2D data loaded, contains {len(self.positions_2d)} subjects")
        print(f"H36M 3D data loaded, contains {len(self.positions_3d)} subjects")
        print(f"Training subjects: {self.train_subjects}")
        print(f"Test subjects: {self.test_subjects}")

        # Validate data alignment
        self._validate_data_alignment()

        # Create dt_dataset structure for compatibility
        self.dt_dataset = {
            'train': {'action': [], '2.5d_factor': [], 'joints_2.5d_image': [], 'source': []},
            'test': {'action': [], '2.5d_factor': [], 'joints_2.5d_image': [], 'source': []}
        }

    def _validate_data_alignment(self):
        """Validate alignment between CPN 2D and H36M 3D data"""
        print("Validating alignment of CPN 2D and H36M 3D data...")

        cpn_subjects = set(self.positions_2d.keys())
        h36m_subjects = set(self.positions_3d.keys())
        common_subjects = cpn_subjects.intersection(h36m_subjects)

        print(f"CPN subjects: {sorted(cpn_subjects)}")
        print(f"H36M subjects: {sorted(h36m_subjects)}")
        print(f"Common subjects: {sorted(common_subjects)}")

        # Verify that training and test subjects exist in H36M data
        missing_train = set(self.train_subjects) - h36m_subjects
        missing_test = set(self.test_subjects) - h36m_subjects
        if missing_train:
            raise ValueError(f"Training subjects missing in H36M 3D data: {missing_train}")
        if missing_test:
            raise ValueError(f"Test subjects missing in H36M 3D data: {missing_test}")

        # Check action alignment
        for subject in self.train_subjects + self.test_subjects:
            if subject in cpn_subjects and subject in h36m_subjects:
                cpn_actions = set(self.positions_2d[subject].keys())
                h36m_actions = set(self.positions_3d[subject].keys())
                common_actions = cpn_actions.intersection(h36m_actions)
                print(f"Subject {subject}: CPN actions={len(cpn_actions)}, H36M actions={len(h36m_actions)}, common actions={len(common_actions)}")
                if len(common_actions) == 0:
                    raise ValueError(f"Subject {subject} has no common actions")

        print("Data alignment validation passed!")

    def prepare_dataset(self):
        """Prepare training and test datasets, build frame-level 2D+3D aligned arrays"""
        if self.gt_trainset is not None and self.gt_testset is not None:
            return

        print("Preparing datasets (building frame-level 2D+3D arrays)...")
        self.gt_trainset = self._prepare_subject_data(self.train_subjects, is_train=True)
        self.gt_testset = self._prepare_subject_data(self.test_subjects, is_train=False)

        print(f"Total training frames: {len(self.gt_trainset['joint_2d'])}")
        print(f"Total test frames: {len(self.gt_testset['joint_2d'])}")

    def _prepare_subject_data(self, subjects, is_train=True):
        """
        Build frame-level 2D+3D arrays for specified subjects, ensuring each frame has corresponding 2D and 3D coordinates
        """
        all_joint_2d = []
        all_confidence = []
        all_camera_names = []
        all_sources = []
        all_joint_3d = []

        dataset_name = "Train" if is_train else "Test"
        print(f"Processing {dataset_name} data, subjects: {subjects}")

        for subject in subjects:
            if subject not in self.positions_2d:
                raise ValueError(f"Subject {subject} not in CPN 2D data")
            if subject not in self.positions_3d:
                raise ValueError(f"Subject {subject} not in H36M 3D data")

            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in subject_2d.items():
                if action_name not in subject_3d:
                    raise ValueError(f"Action {action_name} does not exist in 3D data for subject {subject}")

                # Get the 3D sequence for this action (take first 17 joints)
                full_3d = subject_3d[action_name][:, :17, :]  # [T3d, 17, 3]

                # Process each camera view
                for cam_idx, cam_view_2d in enumerate(action_2d_list):
                    if cam_view_2d is None:
                        continue
                    if cam_idx >= 4:  # Only take the first 4 cameras
                        break

                    camera_name = self.camera_names[cam_idx]
                    full_2d = cam_view_2d.astype(np.float32)  # [T2d, 17, 2]

                    # Ensure 2D and 3D lengths match (take the shorter one)
                    min_len = min(len(full_2d), len(full_3d))
                    if min_len == 0:
                        continue
                    if len(full_2d) != len(full_3d):
                        full_2d = full_2d[:min_len]
                        full_3d = full_3d[:min_len]

                    # Generate confidence (all ones, can be modified later)
                    confidence = np.ones((min_len, 17, 1), dtype=np.float32)

                    # Generate source identifier (one per frame)
                    source_base = f"{subject}_{action_name}_cam{cam_idx}"
                    sources = [source_base] * min_len

                    all_joint_2d.append(full_2d)
                    all_confidence.append(confidence)
                    all_camera_names.extend([camera_name] * min_len)
                    all_sources.extend(sources)
                    all_joint_3d.append(full_3d)  # Add corresponding 3D data

        if len(all_joint_2d) == 0:
            raise ValueError(f"No valid data in {dataset_name} set")

        # Concatenate all arrays
        joint_2d_combined = np.concatenate(all_joint_2d, axis=0)
        confidence_combined = np.concatenate(all_confidence, axis=0)
        joint_3d_combined = np.concatenate(all_joint_3d, axis=0)
        camera_names_array = np.array(all_camera_names)
        sources_array = np.array(all_sources)

        return {
            'joint_2d': joint_2d_combined,
            'confidence': confidence_combined,
            'joint_3d': joint_3d_combined,
            'camera_name': camera_names_array,
            'source': sources_array
        }

    def read_2d(self):
        """Read 2D detection data (normalized)"""
        self.prepare_dataset()

        trainset = self.gt_trainset['joint_2d'][::self.sample_stride]
        testset = self.gt_testset['joint_2d'][::self.sample_stride]

        # Normalize to [-1, 1]
        trainset = self._normalize_2d_data(trainset, is_train=True)
        testset = self._normalize_2d_data(testset, is_train=False)

        if self.read_confidence:
            train_conf = self.gt_trainset['confidence'][::self.sample_stride]
            test_conf = self.gt_testset['confidence'][::self.sample_stride]

            # Ensure confidence shape is correct
            if len(train_conf.shape) == 2:
                train_conf = train_conf[:, :, None]
                test_conf = test_conf[:, :, None]

            # Concatenate as [N, 17, 3]
            trainset = np.concatenate((trainset, train_conf), axis=2)
            testset = np.concatenate((testset, test_conf), axis=2)

        return trainset, testset

    def _normalize_2d_data(self, data, is_train=True):
        """Normalize 2D data to [-1, 1] range based on camera resolution"""
        normalized = data.copy()
        camera_names = self.gt_trainset['camera_name'][::self.sample_stride] if is_train else self.gt_testset['camera_name'][::self.sample_stride]

        for idx, cam in enumerate(camera_names):
            if idx >= len(normalized):
                break
            if cam in ['54138969', '60457274']:
                res_w, res_h = 1000, 1002
            else:  # '55011271', '58860488' or others
                res_w, res_h = 1000, 1000

            normalized[idx, :, 0] = normalized[idx, :, 0] / res_w * 2 - 1
            normalized[idx, :, 1] = normalized[idx, :, 1] / res_w * 2 - res_h / res_w

        return normalized

    def read_3d(self):
        """Read 3D ground truth data (directly return from pre-loaded arrays)"""
        self.prepare_dataset()
        train_labels = self.gt_trainset['joint_3d'][::self.sample_stride]
        test_labels = self.gt_testset['joint_3d'][::self.sample_stride]
        print("Using H36M original 3D ground truth for training and evaluation")
        return train_labels, test_labels

    def read_hw(self):
        """Read image resolution information for test set"""
        if self.test_hw is not None:
            return self.test_hw

        self.prepare_dataset()
        test_cameras = self.gt_testset['camera_name'][::self.sample_stride]
        test_hw = np.zeros((len(test_cameras), 2), dtype=np.float32)

        for idx, cam in enumerate(test_cameras):
            if cam in ['54138969', '60457274']:
                test_hw[idx] = [1000, 1002]
            else:
                test_hw[idx] = [1000, 1000]

        self.test_hw = test_hw
        return test_hw

    def get_split_id(self):
        """Get sliding window split indices and update metadata in dt_dataset"""
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test

        self.prepare_dataset()

        # Get source lists (already with sample_stride applied)
        vid_list_train = self.gt_trainset['source'][::self.sample_stride]
        vid_list_test = self.gt_testset['source'][::self.sample_stride]

        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)

        # Update dt_dataset to match sliced frame indices
        if len(self.split_id_train) > 0:
            # Build action labels per clip, not per frame
            train_clip_actions = []
            train_clip_sources = []

            for clip in self.split_id_train:
                if len(clip) > 0:
                    # Use the first frame of the clip to determine action
                    first_frame_idx = clip[0]
                    if first_frame_idx < len(self.gt_trainset['source']):
                        source = self.gt_trainset['source'][first_frame_idx]
                        action = self._extract_action_from_source(source)
                        train_clip_sources.append(source)
                        train_clip_actions.append(action)

            self.dt_dataset['train']['source'] = train_clip_sources
            self.dt_dataset['train']['action'] = train_clip_actions
            self.dt_dataset['train']['2.5d_factor'] = np.ones(len(train_clip_actions))
            self.dt_dataset['train']['joints_2.5d_image'] = np.zeros((len(train_clip_actions), 17, 3))

        if len(self.split_id_test) > 0:
            # Build action labels per clip, not per frame
            test_clip_actions = []
            test_clip_sources = []

            for clip in self.split_id_test:
                if len(clip) > 0:
                    # Use the first frame of the clip to determine action
                    first_frame_idx = clip[0]
                    if first_frame_idx < len(self.gt_testset['source']):
                        source = self.gt_testset['source'][first_frame_idx]
                        action = self._extract_action_from_source(source)
                        test_clip_sources.append(source)
                        test_clip_actions.append(action)

            self.dt_dataset['test']['source'] = test_clip_sources
            self.dt_dataset['test']['action'] = test_clip_actions
            self.dt_dataset['test']['2.5d_factor'] = np.ones(len(test_clip_actions))
            self.dt_dataset['test']['joints_2.5d_image'] = np.zeros((len(test_clip_actions), 17, 3))

        return self.split_id_train, self.split_id_test

    def _extract_action_from_source(self, source):
        """Extract action name from source string, remove numeric suffix"""
        try:
            # Format: subject_action_camX
            action_with_number = source.split('_')[1]
            # Remove numeric suffix (e.g., 'Directions 1' -> 'Directions')
            base_action = action_with_number.split(' ')[0]

            # Map to standard action names
            standard_actions = {
                'Directions': 'Directions',
                'Discussion': 'Discussion',
                'Eating': 'Eating',
                'Greeting': 'Greeting',
                'Phoning': 'Phoning',
                'Photo': 'Photo',
                'Posing': 'Posing',
                'Purchases': 'Purchases',
                'Sitting': 'Sitting',
                'SittingDown': 'SittingDown',
                'Smoking': 'Smoking',
                'Waiting': 'Waiting',
                'WalkDog': 'WalkDog',
                'Walking': 'Walking',
                'WalkTogether': 'WalkTogether'
            }

            return standard_actions.get(base_action, base_action)
        except:
            return 'unknown'

    def get_hw(self):
        """Get hardware information for test set (for denormalization)"""
        test_hw = self.read_hw()
        _, split_id_test = self.get_split_id()
        if len(split_id_test) > 0:
            # Take hardware info for the first frame of each clip
            clip_hw_indices = [clip[0] for clip in split_id_test]
            test_hw = test_hw[clip_hw_indices]
        return test_hw

    def get_sliced_data(self):
        """Return None (data already saved to disk via save_sliced_data)"""
        self.prepare_dataset()
        return None, None, None, None

    def denormalize(self, test_data):
        """Denormalize data (same as direct version)"""
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])

        if len(data) != len(test_hw):
            print(f"Warning: Prediction data length ({len(data)}) does not match hardware info length ({len(test_hw)})")
            if len(data) > len(test_hw):
                last_hw = test_hw[-1:]
                repeat_times = len(data) - len(test_hw)
                test_hw = np.concatenate([test_hw, np.repeat(last_hw, repeat_times, axis=0)], axis=0)
            else:
                test_hw = test_hw[:len(data)]

        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2

        return data

    def save_sliced_data(self, output_dir):
        """
        Iterate over all subjects and actions, generate sliding window clips and save to disk.
        2D input from CPN, 3D labels from H36M original annotations.
        """
        import pickle
        from tqdm import tqdm
        import os

        print("Generating data clips...")
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        n_frames = self.n_frames
        stride_train = self.data_stride_train
        stride_test = self.data_stride_test
        sample_stride = self.sample_stride

        train_counter = 0
        test_counter = 0

        # Process training set
        for subject in self.train_subjects:
            if subject not in self.positions_2d or subject not in self.positions_3d:
                continue
            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in tqdm(subject_2d.items(), desc=f'Train {subject}'):
                if action_name not in subject_3d:
                    print(f"Warning: Action {action_name} not found in 3D data, skipping")
                    continue

                full_3d = subject_3d[action_name][:, :17, :]  # [T, 17, 3]

                for cam_idx, cam_2d in enumerate(action_2d_list):
                    if cam_2d is None or cam_idx >= 4:
                        continue
                    camera_name = self.camera_names[cam_idx]
                    full_2d = cam_2d.astype(np.float32)  # [T, 17, 2]

                    # Align lengths
                    min_len = min(len(full_2d), len(full_3d))
                    if min_len < n_frames:
                        continue
                    full_2d = full_2d[:min_len]
                    full_3d = full_3d[:min_len]

                    # Apply sample_stride
                    full_2d = full_2d[::sample_stride]
                    full_3d = full_3d[::sample_stride]

                    # Sliding window
                    for start in range(0, len(full_2d) - n_frames + 1, stride_train):
                        clip_input = full_2d[start:start + n_frames]  # [n_frames, 17, 2]
                        clip_label = full_3d[start:start + n_frames]  # [n_frames, 17, 3]
                        confidence = np.ones((n_frames, 17, 1), dtype=np.float32)

                        # Concatenate confidence to input
                        clip_input_with_conf = np.concatenate((clip_input, confidence), axis=2)

                        clip_data = {
                            'data_input': clip_input_with_conf,
                            'data_label': clip_label,
                            'source': f"{subject}_{action_name}_cam{cam_idx}",
                            'camera_name': camera_name,
                            'subject': subject,
                            'action': action_name,
                            'start_frame': start
                        }

                        save_path = os.path.join(train_dir, f"{train_counter:08d}.pkl")
                        with open(save_path, 'wb') as f:
                            pickle.dump(clip_data, f)
                        train_counter += 1

        # Process test set
        for subject in self.test_subjects:
            if subject not in self.positions_2d or subject not in self.positions_3d:
                continue
            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in tqdm(subject_2d.items(), desc=f'Test {subject}'):
                if action_name not in subject_3d:
                    print(f"Warning: Action {action_name} not found in 3D data, skipping")
                    continue

                full_3d = subject_3d[action_name][:, :17, :]

                for cam_idx, cam_2d in enumerate(action_2d_list):
                    if cam_2d is None or cam_idx >= 4:
                        continue
                    camera_name = self.camera_names[cam_idx]
                    full_2d = cam_2d.astype(np.float32)

                    min_len = min(len(full_2d), len(full_3d))
                    if min_len < n_frames:
                        continue
                    full_2d = full_2d[:min_len]
                    full_3d = full_3d[:min_len]

                    full_2d = full_2d[::sample_stride]
                    full_3d = full_3d[::sample_stride]

                    for start in range(0, len(full_2d) - n_frames + 1, stride_test):
                        clip_input = full_2d[start:start + n_frames]
                        clip_label = full_3d[start:start + n_frames]
                        confidence = np.ones((n_frames, 17, 1), dtype=np.float32)
                        clip_input_with_conf = np.concatenate((clip_input, confidence), axis=2)

                        clip_data = {
                            'data_input': clip_input_with_conf,
                            'data_label': clip_label,
                            'source': f"{subject}_{action_name}_cam{cam_idx}",
                            'camera_name': camera_name,
                            'subject': subject,
                            'action': action_name,
                            'start_frame': start
                        }

                        save_path = os.path.join(test_dir, f"{test_counter:08d}.pkl")
                        with open(save_path, 'wb') as f:
                            pickle.dump(clip_data, f)
                        test_counter += 1

        print(f"Data slicing completed! Train clips: {train_counter}, Test clips: {test_counter}")
        print(f"Save path: {output_dir}")