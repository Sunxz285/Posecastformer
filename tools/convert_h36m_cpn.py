import os
import sys
import argparse
sys.path.insert(0, os.getcwd())
from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CPN detection data to training clips (direct version)')
    parser.add_argument('--n_frames', type=int, default=81, help='Number of frames')
    parser.add_argument('--sample_stride', type=int, default=1, help='Sampling stride')
    parser.add_argument('--data_stride_train', type=int, default=27, help='Training data stride')
    parser.add_argument('--data_stride_test', type=int, default=81, help='Test data stride')
    parser.add_argument('--output_dir', type=str, default="data/motion3d/MB3D_f81s27/H36M-CPN",
                       help='Output directory')
    args = parser.parse_args()

    # Create data reader
    datareader = DataReaderH36M_CPN(
        n_frames=args.n_frames,
        sample_stride=args.sample_stride,
        data_stride_train=args.data_stride_train,
        data_stride_test=args.data_stride_test,
        dt_file='data_2d_h36m_cpn_ft_h36m_dbb.npz',
        dt_root='data/motion3d/'
    )

    # Directly save slices to disk
    datareader.save_sliced_data(args.output_dir)

    print("CPN data conversion completed! (direct version)")
    print(f"Data saved to: {args.output_dir}")

