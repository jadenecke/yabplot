import pytest
from yabplot.data import get_available_resources, get_atlas_regions

def test_get_available_resources():
    """Verify that all resource categories exist and contain data."""
    # all categories
    res_all = get_available_resources(None)
    assert isinstance(res_all, dict)

    expected_categories = ['cortical', 'subcortical', 'tracts', 'bmesh']
    for cat in expected_categories:
        assert cat in res_all, f"expected category {cat} to be in resources"
        assert len(res_all[cat]) > 0

    # specific category
    res_cortical = get_available_resources('cortical')
    assert isinstance(res_cortical, list)
    assert 'aparc' in res_cortical

def test_get_atlas_regions_cortical():
    """Verify that cortical atlas regions are correctly retrieved."""
    regions = get_atlas_regions('aparc', 'cortical')
    assert isinstance(regions, list)
    assert len(regions) > 0

def test_get_atlas_regions_subcortical():
    """Verify that subcortical atlas regions are correctly retrieved."""
    regions = get_atlas_regions('aseg', 'subcortical')
    assert isinstance(regions, list)
    assert len(regions) > 0

def test_get_atlas_regions_tracts():
    """Verify that tract atlas regions are correctly retrieved."""
    regions = get_atlas_regions('xtract_tiny', 'tracts')
    assert isinstance(regions, list)
    assert len(regions) > 0

def test_get_atlas_regions_invalid():
    """Verify that invalid categories return an empty list gracefully."""
    regions = get_atlas_regions('aparc', 'invalid_category')
    assert regions == []
