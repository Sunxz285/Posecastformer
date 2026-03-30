import numpy as np
import os
import random
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips

random.seed(0)


class DataReaderH36M_CPN(object):
    """
    CPN检测数据的Human3.6M数据读取器（修改版）
    使用CPN 2D检测结果 + H36M原始3D标注，严格对齐，不使用SH数据
    """

    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True,
                 dt_root='data/motion3d', dt_file='data_2d_h36m_cpn_ft_h36m_dbb.npz'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_root = dt_root

        # 加载CPN 2D数据
        cpn_file_path = os.path.join(dt_root, dt_file)
        print(f"加载CPN 2D数据: {cpn_file_path}")
        self.cpn_data = np.load(cpn_file_path, allow_pickle=True)
        self.positions_2d = self.cpn_data['positions_2d'].item()
        self.metadata = self.cpn_data['metadata'].item()

        # 加载H36M 3D真值数据
        h36m_3d_path = os.path.join(dt_root, 'data_3d_h36m.npz')
        print(f"加载H36M 3D真值数据: {h36m_3d_path}")
        self.h36m_3d_data = np.load(h36m_3d_path, allow_pickle=True)
        self.positions_3d = self.h36m_3d_data['positions_3d'].item()

        # 定义训练/测试集划分
        self.train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_subjects = ['S9', 'S11']

        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

        # 摄像机名称 (4个视角)
        self.camera_names = ['54138969', '60457274', '55011271', '58860488']

        print(f"CPN 2D数据加载完成，包含 {len(self.positions_2d)} 个受试者")
        print(f"H36M 3D数据加载完成，包含 {len(self.positions_3d)} 个受试者")
        print(f"训练受试者: {self.train_subjects}")
        print(f"测试受试者: {self.test_subjects}")

        # 验证数据对齐
        self._validate_data_alignment()

        # 为兼容性创建dt_dataset结构
        self.dt_dataset = {
            'train': {'action': [], '2.5d_factor': [], 'joints_2.5d_image': [], 'source': []},
            'test': {'action': [], '2.5d_factor': [], 'joints_2.5d_image': [], 'source': []}
        }

    def _validate_data_alignment(self):
        """验证CPN 2D和H36M 3D数据的对齐性"""
        print("验证CPN 2D和H36M 3D数据对齐性...")

        cpn_subjects = set(self.positions_2d.keys())
        h36m_subjects = set(self.positions_3d.keys())
        common_subjects = cpn_subjects.intersection(h36m_subjects)

        print(f"CPN受试者: {sorted(cpn_subjects)}")
        print(f"H36M受试者: {sorted(h36m_subjects)}")
        print(f"共同受试者: {sorted(common_subjects)}")

        # 验证训练和测试受试者都在H36M数据中
        missing_train = set(self.train_subjects) - h36m_subjects
        missing_test = set(self.test_subjects) - h36m_subjects
        if missing_train:
            raise ValueError(f"训练受试者在H36M 3D数据中缺失: {missing_train}")
        if missing_test:
            raise ValueError(f"测试受试者在H36M 3D数据中缺失: {missing_test}")

        # 检查动作对齐
        for subject in self.train_subjects + self.test_subjects:
            if subject in cpn_subjects and subject in h36m_subjects:
                cpn_actions = set(self.positions_2d[subject].keys())
                h36m_actions = set(self.positions_3d[subject].keys())
                common_actions = cpn_actions.intersection(h36m_actions)
                print(f"受试者 {subject}: CPN动作数={len(cpn_actions)}, H36M动作数={len(h36m_actions)}, 共同动作数={len(common_actions)}")
                if len(common_actions) == 0:
                    raise ValueError(f"受试者 {subject} 没有共同的动作")

        print("数据对齐验证通过!")

    def prepare_dataset(self):
        """准备训练和测试数据集，构建2D+3D对齐的帧级数组"""
        if self.gt_trainset is not None and self.gt_testset is not None:
            return

        print("准备数据集（构建帧级2D+3D数组）...")
        self.gt_trainset = self._prepare_subject_data(self.train_subjects, is_train=True)
        self.gt_testset = self._prepare_subject_data(self.test_subjects, is_train=False)

        print(f"训练集总帧数: {len(self.gt_trainset['joint_2d'])}")
        print(f"测试集总帧数: {len(self.gt_testset['joint_2d'])}")

    def _prepare_subject_data(self, subjects, is_train=True):
        """
        为指定受试者构建帧级2D+3D数组，确保每帧都有对应的2D和3D坐标
        """
        all_joint_2d = []
        all_confidence = []
        all_camera_names = []
        all_sources = []
        all_joint_3d = []

        dataset_name = "训练" if is_train else "测试"
        print(f"处理{dataset_name}数据，受试者: {subjects}")

        for subject in subjects:
            if subject not in self.positions_2d:
                raise ValueError(f"受试者 {subject} 不在CPN 2D数据中")
            if subject not in self.positions_3d:
                raise ValueError(f"受试者 {subject} 不在H36M 3D数据中")

            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in subject_2d.items():
                if action_name not in subject_3d:
                    raise ValueError(f"动作 {action_name} 在受试者 {subject} 的3D数据中不存在")

                # 获取该动作的3D序列 (取前17个关节)
                full_3d = subject_3d[action_name][:, :17, :]  # [T3d, 17, 3]

                # 处理每个摄像头视角
                for cam_idx, cam_view_2d in enumerate(action_2d_list):
                    if cam_view_2d is None:
                        continue
                    if cam_idx >= 4:  # 只取前4个摄像机
                        break

                    camera_name = self.camera_names[cam_idx]
                    full_2d = cam_view_2d.astype(np.float32)  # [T2d, 17, 2]

                    # 确保2D和3D长度一致（取较短者）
                    min_len = min(len(full_2d), len(full_3d))
                    if min_len == 0:
                        continue
                    if len(full_2d) != len(full_3d):
                        full_2d = full_2d[:min_len]
                        full_3d = full_3d[:min_len]

                    # 生成置信度（全1，可后续修改）
                    confidence = np.ones((min_len, 17, 1), dtype=np.float32)

                    # 生成source标识符（每帧一个）
                    source_base = f"{subject}_{action_name}_cam{cam_idx}"
                    sources = [source_base] * min_len

                    all_joint_2d.append(full_2d)
                    all_confidence.append(confidence)
                    all_camera_names.extend([camera_name] * min_len)
                    all_sources.extend(sources)
                    all_joint_3d.append(full_3d)  # 添加对应的3D数据

        if len(all_joint_2d) == 0:
            raise ValueError(f"{dataset_name}集没有有效数据")

        # 合并所有数组
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
        """读取2D检测数据（已归一化）"""
        self.prepare_dataset()

        trainset = self.gt_trainset['joint_2d'][::self.sample_stride]
        testset = self.gt_testset['joint_2d'][::self.sample_stride]

        # 归一化到[-1, 1]
        trainset = self._normalize_2d_data(trainset, is_train=True)
        testset = self._normalize_2d_data(testset, is_train=False)

        if self.read_confidence:
            train_conf = self.gt_trainset['confidence'][::self.sample_stride]
            test_conf = self.gt_testset['confidence'][::self.sample_stride]

            # 确保置信度形状正确
            if len(train_conf.shape) == 2:
                train_conf = train_conf[:, :, None]
                test_conf = test_conf[:, :, None]

            # 合并为 [N, 17, 3]
            trainset = np.concatenate((trainset, train_conf), axis=2)
            testset = np.concatenate((testset, test_conf), axis=2)

        return trainset, testset

    def _normalize_2d_data(self, data, is_train=True):
        """归一化2D数据到[-1, 1]范围，基于摄像机分辨率"""
        normalized = data.copy()
        camera_names = self.gt_trainset['camera_name'][::self.sample_stride] if is_train else self.gt_testset['camera_name'][::self.sample_stride]

        for idx, cam in enumerate(camera_names):
            if idx >= len(normalized):
                break
            if cam in ['54138969', '60457274']:
                res_w, res_h = 1000, 1002
            else:  # '55011271', '58860488' 或其他
                res_w, res_h = 1000, 1000

            normalized[idx, :, 0] = normalized[idx, :, 0] / res_w * 2 - 1
            normalized[idx, :, 1] = normalized[idx, :, 1] / res_w * 2 - res_h / res_w

        return normalized

    def read_3d(self):
        """读取3D ground truth数据（直接从预加载的数组中返回）"""
        self.prepare_dataset()
        train_labels = self.gt_trainset['joint_3d'][::self.sample_stride]
        test_labels = self.gt_testset['joint_3d'][::self.sample_stride]
        print("使用H36M原始3D ground truth进行训练和评估")
        return train_labels, test_labels

    def read_hw(self):
        """读取测试集的图像分辨率信息"""
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
        """获取滑窗切分索引，并更新dt_dataset中的元数据"""
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test

        self.prepare_dataset()

        # 获取source列表（已应用sample_stride）
        vid_list_train = self.gt_trainset['source'][::self.sample_stride]
        vid_list_test = self.gt_testset['source'][::self.sample_stride]

        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)

        # 更新dt_dataset以匹配切分后的帧索引
        if len(self.split_id_train) > 0:
            # 按 clip 构建动作标签，而不是按帧
            train_clip_actions = []
            train_clip_sources = []

            for clip in self.split_id_train:
                if len(clip) > 0:
                    # 使用clip的第一个帧来确定动作
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
            # 按 clip 构建动作标签，而不是按帧
            test_clip_actions = []
            test_clip_sources = []

            for clip in self.split_id_test:
                if len(clip) > 0:
                    # 使用clip的第一个帧来确定动作
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
        """从source字符串中提取动作名，去除数字后缀"""
        try:
            # 格式: subject_action_camX
            action_with_number = source.split('_')[1]
            # 去除数字后缀（如 'Directions 1' -> 'Directions'）
            base_action = action_with_number.split(' ')[0]

            # 映射到标准动作名称
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
        """获取测试集的硬件信息（用于反归一化）"""
        test_hw = self.read_hw()
        _, split_id_test = self.get_split_id()
        if len(split_id_test) > 0:
            # 取每个clip的第一帧对应的硬件信息
            clip_hw_indices = [clip[0] for clip in split_id_test]
            test_hw = test_hw[clip_hw_indices]
        return test_hw

    def get_sliced_data(self):
        """返回None（数据已通过save_sliced_data保存到磁盘）"""
        self.prepare_dataset()
        return None, None, None, None

    def denormalize(self, test_data):
        """反归一化数据（与direct版本相同）"""
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])

        if len(data) != len(test_hw):
            print(f"警告: 预测数据长度({len(data)})与硬件信息长度({len(test_hw)})不匹配")
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
        遍历所有受试者和动作，生成滑窗片段并保存到磁盘。
        2D输入来自CPN，3D标签来自H36M原始标注。
        """
        import pickle
        from tqdm import tqdm
        import os

        print("开始生成数据片段...")
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

        # 处理训练集
        for subject in self.train_subjects:
            if subject not in self.positions_2d or subject not in self.positions_3d:
                continue
            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in tqdm(subject_2d.items(), desc=f'训练 {subject}'):
                if action_name not in subject_3d:
                    print(f"警告: 动作 {action_name} 在3D数据中不存在，跳过")
                    continue

                full_3d = subject_3d[action_name][:, :17, :]  # [T, 17, 3]

                for cam_idx, cam_2d in enumerate(action_2d_list):
                    if cam_2d is None or cam_idx >= 4:
                        continue
                    camera_name = self.camera_names[cam_idx]
                    full_2d = cam_2d.astype(np.float32)  # [T, 17, 2]

                    # 长度对齐
                    min_len = min(len(full_2d), len(full_3d))
                    if min_len < n_frames:
                        continue
                    full_2d = full_2d[:min_len]
                    full_3d = full_3d[:min_len]

                    # 应用sample_stride
                    full_2d = full_2d[::sample_stride]
                    full_3d = full_3d[::sample_stride]

                    # 滑窗
                    for start in range(0, len(full_2d) - n_frames + 1, stride_train):
                        clip_input = full_2d[start:start + n_frames]  # [n_frames, 17, 2]
                        clip_label = full_3d[start:start + n_frames]  # [n_frames, 17, 3]
                        confidence = np.ones((n_frames, 17, 1), dtype=np.float32)

                        # 合并置信度到输入
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

        # 处理测试集
        for subject in self.test_subjects:
            if subject not in self.positions_2d or subject not in self.positions_3d:
                continue
            subject_2d = self.positions_2d[subject]
            subject_3d = self.positions_3d[subject]

            for action_name, action_2d_list in tqdm(subject_2d.items(), desc=f'测试 {subject}'):
                if action_name not in subject_3d:
                    print(f"警告: 动作 {action_name} 在3D数据中不存在，跳过")
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

        print(f"数据切片完成！训练片段: {train_counter}, 测试片段: {test_counter}")
        print(f"保存路径: {output_dir}")