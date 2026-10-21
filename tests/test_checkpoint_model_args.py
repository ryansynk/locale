"""The benchmark rebuilds LOCALE from a checkpoint's model_args
(src/encoders.py locale_init_args): a projector-trained checkpoint must come
back with its projector, whether or not the flag was recorded."""

from lae.modeling.backbones import DEFAULT_BACKBONE
from src.encoders import locale_init_args

BASE = {"pooling": "mean", "dim": 128, "K": 0, "m": 0.9995, "T": 0.05}


def test_recorded_flag_wins():
    ckpt = {
        "model_args": {**BASE, "use_projection_head": True, "backbone": "nt50m"},
        "model": {},
    }
    args = locale_init_args(ckpt)
    assert args["use_projection_head"] is True
    assert args["backbone"] == "nt50m"
    assert args["K"] == 0 and args["pooling"] == "mean"


def test_old_checkpoint_infers_projector_from_weights():
    with_proj = {"model_args": dict(BASE), "model": {"projector_q.0.weight": 0}}
    without = {"model_args": dict(BASE), "model": {"bert_q.x": 0}}
    assert locale_init_args(with_proj)["use_projection_head"] is True
    assert locale_init_args(without)["use_projection_head"] is False
    assert locale_init_args(without)["backbone"] == DEFAULT_BACKBONE
