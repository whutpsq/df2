from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


class BevFusionRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0") -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        self.config_path = str(Path(config_path).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_config(self.config_path)
        self.model = self._build_model()

    def infer(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = self._move_to_device(data)
        with torch.inference_mode():
            return self.model(return_loss=False, rescale=True, **data)

    def _move_to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list) and value and torch.is_tensor(value[0]):
                moved[key] = [item.to(self.device, non_blocking=True) for item in value]
            else:
                moved[key] = value
        return moved

    def _load_config(self, config_path: str):
        from mmcv import Config
        from torchpack.utils.config import configs
        from mmdet3d.utils import recursive_eval

        configs.load(config_path, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        return cfg

    def _build_model(self):
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet3d.models import build_model

        torch.backends.cudnn.benchmark = True
        model = build_model(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.checkpoint_path, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        elif hasattr(self.cfg, "object_classes"):
            model.CLASSES = self.cfg.object_classes
        model.to(self.device)
        model.eval()
        return model
