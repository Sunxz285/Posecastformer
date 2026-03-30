# test_flops_latency.py
import os
import sys
import argparse
import time
import logging
from datetime import datetime
import torch
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from thop import profile, clever_format

def setup_logging(log_file=None):
    """Configure logging, output to both console and file (if specified), file mode is append ('a')"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers (if logger already exists)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is specified), use append mode
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Test FLOPs and Latency of PoseCASTformer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.bin)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of iterations for latency measurement')
    parser.add_argument('--warmup-iterations', type=int, default=10, help='Warmup iterations before timing')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path (appended if exists)')
    return parser.parse_args()

def load_model(args, config, logger):
    logger.info(f"Building model with backbone: {config.backbone}")
    model = load_backbone(config)

    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_pos' in checkpoint:
        state_dict = checkpoint['model_pos']
        logger.info("Checkpoint contains 'model_pos' key, using it.")
    else:
        state_dict = checkpoint
        logger.info("Checkpoint loaded directly as state dict.")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info("Model loaded successfully.")
    return model

def main():
    args = parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("FLOPs & Latency Test Started")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Checkpoint file: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of iterations: {args.num_iterations}")
    logger.info(f"Warmup iterations: {args.warmup_iterations}")

    try:
        cfg = get_config(args.config)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    if not hasattr(cfg, 'no_conf'):
        cfg.no_conf = False

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    logger.info(f"Using device: {device}")

    model = load_model(args, cfg, logger)
    model = model.to(device)
    model.eval()

    batch_size = args.batch_size
    T = cfg.maxlen
    J = cfg.num_joints
    C = 2 if cfg.no_conf else 3
    input_tensor = torch.randn(batch_size, T, J, C).to(device)
    logger.info(f"Input shape: {input_tensor.shape} (B, T, J, C)")

    # FLOPs
    logger.info("Calculating FLOPs and parameters...")
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        logger.info(f"FLOPs: {flops}  |  Parameters: {params}")
    except Exception as e:
        logger.error(f"FLOPs calculation failed: {e}")

    # Latency
    logger.info(f"Measuring latency with {args.num_iterations} iterations...")
    logger.info(f"Warming up for {args.warmup_iterations} iterations...")
    for i in range(args.warmup_iterations):
        _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if (i + 1) % 10 == 0:
            logger.debug(f"Warmup iteration {i+1}/{args.warmup_iterations}")

    logger.info("Starting timing...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for i in range(args.num_iterations):
        _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if (i + 1) % 20 == 0:
            logger.debug(f"Progress: {i+1}/{args.num_iterations}")
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = total_time / args.num_iterations * 1000
    logger.info(f"Total time for {args.num_iterations} iterations: {total_time:.3f} s")
    logger.info(f"Average inference time per iteration (batch size {batch_size}): {avg_time_ms:.3f} ms")

    logger.info("=" * 60)
    logger.info("Test completed.\n")  # Additional blank line to separate runs

if __name__ == '__main__':
    main()