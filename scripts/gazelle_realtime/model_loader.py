#!/usr/bin/env python3
"""
Model loading utilities for GazeLLE application.
"""
import torch
from pathlib import Path

try:
    from gazelle.model import GazeLLE
    from gazelle.backbone import DinoV2Backbone
except ImportError:
    # Try alternative path if gazelle package not found
    import sys
    current_dir = Path(__file__).parent
    scripts_dir = current_dir.parent
    project_root = scripts_dir.parent
    sys.path.append(str(project_root / "gazelle"))
    from model import GazeLLE
    from backbone import DinoV2Backbone

from utils import get_hef_input_dimensions


def load_gazelle_model(pth_path, hef_path, device='cpu'):
    """Load and configure GazeLLE model."""
    try:
        import hailo_platform as hpf
        
        # Validate file paths
        if not Path(pth_path).exists():
            raise FileNotFoundError(f"PTH file not found: {pth_path}")
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
        
        # Get HEF dimensions
        hef_model = hpf.HEF(hef_path)
        hef_h, hef_w = get_hef_input_dimensions(hef_model)
        print(f"[MODEL] HEF input dimensions: {hef_h}x{hef_w}")
        
        # Create model
        backbone = DinoV2Backbone("dinov2_vits14")
        gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
        
        # Load checkpoint
        print(f"[MODEL] Loading checkpoint: {pth_path}")
        checkpoint = torch.load(pth_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load weights (backbone excluded - using Hailo)
        gazelle_model.load_gazelle_state_dict(state_dict, include_backbone=False)
        gazelle_model.to(device)
        gazelle_model.eval()
        
        print(f"[MODEL] GazeLLE model loaded successfully on device: {device}")
        return gazelle_model
        
    except Exception as e:
        print(f"[ERROR] Failed to load GazeLLE model: {e}")
        raise