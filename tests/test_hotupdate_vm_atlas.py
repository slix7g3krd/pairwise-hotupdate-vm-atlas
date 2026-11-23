import pytest
import gameplay.hotupdate_vm_atlas as mod

CASES = [
    ("closure_generation", (1)),
    ("verify_gate", (1)),
    ("scene_isolate", (1)),
    ("rollback_clear", (1)),
    ("ab_version", (1)),
    ("old_fallback", (1)),
    ("atlas_ref", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("download_state", (1)),
    ("metrics", (1)),
    ("mid_battle", (1)),
    ("cache_key", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
