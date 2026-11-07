import copy

import pytest
import yaml

from conftest import CONFIG_YAML
from frenet_planner.params import Params


def test_config_yaml_loads():
    p = Params.from_yaml(CONFIG_YAML)
    assert p.vehicle.kappa_max == pytest.approx(1.14, abs=0.01)


@pytest.mark.parametrize('mutate', ['typo', 'missing'])
def test_bad_keys_fail_loudly(mutate):
    with open(CONFIG_YAML) as f:
        d = copy.deepcopy(yaml.safe_load(f)['frenet_planner']['ros__parameters'])
    if mutate == 'typo':
        d['vehicle']['wheel_base'] = d['vehicle'].pop('wheelbase')
    else:
        del d['speed']['global_scale']
    with pytest.raises(ValueError):
        Params.from_dict(d)
