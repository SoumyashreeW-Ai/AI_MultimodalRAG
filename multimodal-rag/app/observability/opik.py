from __future__ import annotations
import os
import json
from typing import Optional

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None

_exp: Optional["Experiment"] = None


def init_opik():
    global _exp
    api_key = os.getenv("COMET_API_KEY")
    if not api_key or Experiment is None:
        return None
    _exp = Experiment(api_key=api_key, project_name="multimodal-rag", auto_param_logging=False)
    return _exp


def get_opik():
    return _exp


def log_metric(name: str, value, step: int | None = None):
    if _exp:
        _exp.log_metric(name, value, step=step)


def log_text(name: str, text: str):
    if _exp:
        _exp.log_text(name, text)


def log_params(params: dict):
    if _exp:
        _exp.log_parameters(params)


def safe_redact_text(s: str, max_len: int = 500):
    if not s:
        return ""
    return s[:max_len]
