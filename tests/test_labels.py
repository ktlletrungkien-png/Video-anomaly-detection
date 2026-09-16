import pytest

from rwf2000.labels import UnknownClassError, canonical_label_name, label_from_class_name


@pytest.mark.parametrize("name", ["NonFight", "nonfight", "NonViolence", "Non-violence", "NON-VIOLENCE"])
def test_non_violence_aliases_map_to_zero(name):
    assert label_from_class_name(name) == 0
    assert canonical_label_name(name) == "Non-violence"


@pytest.mark.parametrize("name", ["Fight", "fight", "Violence", "VIOLENCE"])
def test_violence_aliases_map_to_one(name):
    assert label_from_class_name(name) == 1
    assert canonical_label_name(name) == "Violence"


def test_unknown_class_and_nonfight_substring_are_rejected():
    with pytest.raises(UnknownClassError, match="Unknown RWF-2000 class directory"):
        label_from_class_name("NonFighting")
    with pytest.raises(UnknownClassError):
        label_from_class_name("FightExtra")
