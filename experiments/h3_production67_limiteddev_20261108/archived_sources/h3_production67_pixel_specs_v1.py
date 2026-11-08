"""CPU-only pixel-region proposals for explicit H3 production67 geometry.

This is not a PixelSupervisor and cannot decode, backpropagate, or train.
assembled_all recipes cover all226 times symbolically; the previous-style
nonblend_only selection still covers only166. Adopting blend supervision is a
separate objective change, not an automatic part of this geometry helper.
"""
from __future__ import annotations

import copy
import math

from h3_production67_geometry_v1 import (Box, Geometry, canonical_sha, frame_recipe,
                                       integer, temporal_block_frames)

SPEC_SCHEMA = 'h3-production67-visible-pixel-spec-v1-cpu-proposal'


def split_tiles(length, tile_size=256, min_overlap=64, compression=16):
    """Pure arithmetic equivalent of the released VAE tile split formula."""
    for name, value in [('length', length), ('tile_size', tile_size),
                        ('min_overlap', min_overlap), ('compression', compression)]:
        integer(value, name, 1)
    if min_overlap >= tile_size or any(v % compression for v in (length, tile_size, min_overlap)):
        raise ValueError('Require aligned dimensions and overlap smaller than tile')
    if tile_size >= length:
        return [0], [length], []
    count = math.ceil(length / tile_size)
    while tile_size * count - min_overlap * (count - 1) < length:
        count += 1
    overlaps = [min_overlap] * (count - 1)
    slack = tile_size * count - sum(overlaps) - length
    for i in range(slack // compression):
        overlaps[i % (count - 1)] += compression
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts, [tile_size] * count, overlaps


def spatial_regions(geometry: Geometry, *, min_spatial=64):
    integer(min_spatial, 'min_spatial', 1)
    if not isinstance(geometry, Geometry):
        raise ValueError('Require explicit Geometry')
    ys, hs, yo = split_tiles(geometry.raw_hw[0])
    xs, ws, xo = split_tiles(geometry.raw_hw[1])
    result = []
    for row, (y, h) in enumerate(zip(ys, hs)):
        a, b = yo[row - 1] if row else 0, h - (yo[row] if row < len(ys) - 1 else 0)
        for col, (x, w) in enumerate(zip(xs, ws)):
            c, d = xo[col - 1] if col else 0, w - (xo[col] if col < len(xs) - 1 else 0)
            if a >= b or c >= d:
                continue
            raw_global = Box(y + a, y + b, x + c, x + d)
            visible = geometry.intersect_raw_with_visible(raw_global)
            if visible is None or min(visible.y1 - visible.y0, visible.x1 - visible.x0) < min_spatial:
                continue
            cropped_raw = Box(visible.y0 + 4, visible.y1 + 4, visible.x0, visible.x1)
            result.append({'spatial_mode': 'stable_tile_interior', 'tile_row': row, 'tile_column': col,
                'target_latent_yx': [y // 16, (y + h) // 16, x // 16, (x + w) // 16],
                'decoder_raw_tile_origin_yx': [y, x],
                'decoder_raw_tile_hw': [h, w],
                'decoder_local_box_yx': [cropped_raw.y0 - y, cropped_raw.y1 - y,
                                         cropped_raw.x0 - x, cropped_raw.x1 - x],
                'raw_global_box_yx': cropped_raw.as_list(),
                'visible_global_box_yx': visible.as_list(),
                'lr_geometry_enclosure': geometry.visible_box_to_lr_cover(visible)})
    return tuple(result)


def full_spatial_region(geometry):
    visible = Box(0, 1080, 0, 1920)
    return {'spatial_mode': 'full_spatial', 'tile_row': -1, 'tile_column': -1,
            'target_latent_yx': [0, 68, 0, 120],
            'decoder_raw_tile_origin_yx': [0, 0], 'decoder_raw_tile_hw': [1088, 1920],
            'decoder_local_box_yx': [4, 1084, 0, 1920],
            'raw_global_box_yx': [4, 1084, 0, 1920],
            'visible_global_box_yx': visible.as_list(),
            'lr_geometry_enclosure': geometry.visible_box_to_lr_cover(visible)}


class PixelSpecCatalog:
    """Deterministic, JSON-safe, geometry-bound proposed region specs."""
    def __init__(self, geometry, *, temporal_scope, min_spatial=64, lpips_count=4):
        if not isinstance(geometry, Geometry):
            raise ValueError('Require explicit Geometry')
        if temporal_scope not in ('assembled_all', 'nonblend_only'):
            raise ValueError('Temporal scope must be explicit')
        integer(lpips_count, 'lpips_count', 1)
        self.geometry = geometry
        self.temporal_scope = temporal_scope
        self.min_spatial = integer(min_spatial, 'min_spatial', 1)
        self.lpips_count = lpips_count
        self._spatial = spatial_regions(geometry, min_spatial=min_spatial)
        if not self._spatial:
            raise ValueError('No eligible spatial tile interiors')
        self._spatial_lookup = {(s['tile_row'], s['tile_column']): s for s in self._spatial}

    def spec(self, block, *, tile=None):
        frames = temporal_block_frames(block, scope=self.temporal_scope)
        if tile is None:
            spatial = full_spatial_region(self.geometry)
        else:
            if (type(tile) is not tuple or len(tile) != 2 or
                    any(type(i) is not int for i in tile) or tile not in self._spatial_lookup):
                raise ValueError('Unknown tile coordinates')
            spatial = copy.deepcopy(self._spatial_lookup[tile])
        count = min(self.lpips_count, len(frames))
        selected = [0] if count == 1 else [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
        return {'schema': SPEC_SCHEMA, 'geometry_sha256': self.geometry.sha256,
                'temporal_scope': self.temporal_scope, 'block_index': block,
                'rgb_frame_indices': list(frames),
                'temporal_assembly': [frame_recipe(f) for f in frames],
                'lpips_packed_positions': selected,
                'spatial': spatial,
                'actual_decode_or_gradient_validated': False,
                'training_objective_adopted': False}

    def validate_spec(self, spec):
        if type(spec) is not dict:
            raise ValueError('Spec must be a dictionary')
        try:
            spatial = spec['spatial']
            tile = None if spatial['spatial_mode'] == 'full_spatial' else (spatial['tile_row'], spatial['tile_column'])
            expected = self.spec(spec['block_index'], tile=tile)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Unknown or malformed spec') from error
        # Canonical JSON comparison also rejects bool masquerading as int.
        if canonical_sha(spec) != canonical_sha(expected):
            raise ValueError('Stale, incomplete, or mutated spec; rebuild from this catalog')
        return True

    def full_specs(self):
        return tuple(self.spec(block) for block in range(13))

    def tile_specs(self):
        return tuple(self.spec(block, tile=tile) for block in range(13)
                     for tile in self._spatial_lookup)

    def description(self):
        frames = sorted({f for block in range(13) for f in temporal_block_frames(block, scope=self.temporal_scope)})
        return {'schema': SPEC_SCHEMA, 'geometry_sha256': self.geometry.sha256,
                'temporal_scope': self.temporal_scope, 'eligible_frames': frames,
                'excluded_frames': sorted(set(range(226)) - set(frames)),
                'spatial_tiles': len(self._spatial), 'full_specs': 13,
                'tile_specs': 13 * len(self._spatial), 'lpips_count': self.lpips_count,
                'spatial_sampling_is_pixel_unbiased': False,
                'tile_enclosures_are_model_receptive_fields': False,
                'formal_training_allowed': False, 'model_validation_required': True}
