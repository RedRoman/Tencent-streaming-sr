"""CPU-only geometry proposals for H3 T67, 44x80 -> 68x120.

No model, tensor framework, file reader, device, or training entry point is
imported. This module describes coordinates and released-decoder formulas;
passing its checks does not establish model parity, gradients, or trainability.
All boxes are half-open pixel-edge coordinates. Center coordinates are named
explicitly. A mapped input box is NOT a receptive-field or safe model-crop claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

SCHEMA = 'h3-production67-explicit-geometry-v1-cpu-proposal'


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def rational(value):
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    raise ValueError('Use exact integer or Fraction coordinates, not floats')


def fraction_json(value):
    value = rational(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Box:
    """Integer half-open box in a caller-specified coordinate domain."""
    y0: int
    y1: int
    x0: int
    x1: int

    def __post_init__(self):
        for key in ('y0', 'y1', 'x0', 'x1'):
            integer(getattr(self, key), key)
        if self.y0 >= self.y1 or self.x0 >= self.x1:
            raise ValueError('Box must have positive height and width')

    def within(self, height, width):
        if self.y1 > height or self.x1 > width:
            raise ValueError('Box outside its declared coordinate domain')
        return self

    def as_list(self):
        return [self.y0, self.y1, self.x0, self.x1]


@dataclass(frozen=True)
class Geometry:
    """Explicit shape contract; no implicit two-times target or resize fallback."""
    input_bcthw: tuple
    target_bcthw: tuple
    visible_crop: tuple
    rgb_frames: int
    fps: tuple

    def __post_init__(self):
        for shape in (self.input_bcthw, self.target_bcthw):
            if type(shape) is not tuple or len(shape) != 5:
                raise ValueError('Require explicit BCTHW tuples')
            for dim in shape:
                integer(dim, 'shape dimension', 1)
        if self.input_bcthw != (1, 24, 67, 44, 80):
            raise ValueError('This reviewed proposal requires LR [1,24,67,44,80]')
        if self.target_bcthw != (1, 24, 67, 68, 120):
            raise ValueError('Explicit target must be [1,24,67,68,120], not 2x')
        if (type(self.visible_crop) is not tuple or self.visible_crop != (4, 1084, 0, 1920)
                or any(type(x) is not int for x in self.visible_crop)):
            raise ValueError('Visible crop must be raw rows4:1084 and columns0:1920')
        if type(self.rgb_frames) is not int or self.rgb_frames != 226:
            raise ValueError('All226 real RGB frames are required')
        if (type(self.fps) is not tuple or self.fps != (24, 1)
                or any(type(x) is not int for x in self.fps)):
            raise ValueError('Use exact24/1 fps')

    @property
    def input_hw(self):
        return (704, 1280)

    @property
    def raw_hw(self):
        return (1088, 1920)

    @property
    def visible_hw(self):
        return (1080, 1920)

    @property
    def scale_hw(self):
        return (Fraction(68, 44), Fraction(120, 80))

    def forward_kwargs(self):
        """Matches CausalUpscalerAdapter.native_forward's explicit target API.

        The adapter computes the existing scalar conditioning itself. No new
        'scale_factor' keyword is passed to its API, and normalization remains
        the caller/adapter's responsibility. This method invokes no model.
        """
        return {'target_size': self.target_bcthw[2:]}

    @property
    def scalar_condition(self):
        return sum(self.scale_hw, Fraction(0)) / 2

    def description(self):
        return {'schema': SCHEMA, 'input_bcthw': list(self.input_bcthw),
                'target_bcthw': list(self.target_bcthw),
                'input_compute_hw': list(self.input_hw), 'raw_output_hw': list(self.raw_hw),
                'visible_output_hw': list(self.visible_hw),
                'visible_crop_y0_y1_x0_x1': list(self.visible_crop),
                'rgb_frames': self.rgb_frames, 'fps': list(self.fps),
                'scale_h': fraction_json(self.scale_hw[0]),
                'scale_w': fraction_json(self.scale_hw[1]),
                'upstream_scalar_condition': fraction_json(self.scalar_condition),
                'coordinate_convention': 'half-open edges; exact rational affine maps',
                'formal_training_allowed': False, 'actual_model_or_gradient_validation': False}

    @property
    def sha256(self):
        return canonical_sha(self.description())

    def raw_edge_to_lr(self, value, axis):
        value = rational(value)
        axis = checked_axis(axis)
        if not 0 <= value <= self.raw_hw[axis]:
            raise ValueError('Raw edge outside canvas')
        return value / self.scale_hw[axis]

    def lr_edge_to_raw(self, value, axis):
        value = rational(value)
        axis = checked_axis(axis)
        if not 0 <= value <= self.input_hw[axis]:
            raise ValueError('LR edge outside canvas')
        return value * self.scale_hw[axis]

    def visible_edge_to_lr(self, value, axis):
        value = rational(value)
        axis = checked_axis(axis)
        if not 0 <= value <= self.visible_hw[axis]:
            raise ValueError('Visible edge outside crop')
        return self.raw_edge_to_lr(value + (4 if axis == 0 else 0), axis)

    def visible_center_to_lr_center_index(self, pixel_index, axis):
        axis = checked_axis(axis)
        integer(pixel_index, 'pixel index')
        if pixel_index >= self.visible_hw[axis]:
            raise ValueError('Visible pixel index outside crop')
        return self.visible_edge_to_lr(Fraction(pixel_index) + Fraction(1, 2), axis) - Fraction(1, 2)

    def visible_box_to_lr_cover(self, box):
        box.within(*self.visible_hw)
        edges = [self.visible_edge_to_lr(box.y0, 0), self.visible_edge_to_lr(box.y1, 0),
                 self.visible_edge_to_lr(box.x0, 1), self.visible_edge_to_lr(box.x1, 1)]
        cover = Box(math.floor(edges[0]), math.ceil(edges[1]),
                    math.floor(edges[2]), math.ceil(edges[3])).within(*self.input_hw)
        return {'lr_exact_edges_y0_y1_x0_x1': [fraction_json(v) for v in edges],
                'lr_outward_cover_y0_y1_x0_x1': cover.as_list(),
                'meaning': 'Geometric enclosure only; not an interpolation footprint or model receptive field'}

    def intersect_raw_with_visible(self, box):
        box.within(*self.raw_hw)
        y0, y1 = max(box.y0, 4), min(box.y1, 1084)
        if y0 >= y1:
            return None
        return Box(y0 - 4, y1 - 4, box.x0, box.x1)


def production67():
    return Geometry((1, 24, 67, 44, 80), (1, 24, 67, 68, 120),
                    (4, 1084, 0, 1920), 226, (24, 1))


def checked_axis(axis):
    if type(axis) is not int or axis not in (0, 1):
        raise ValueError('Axis must be0(y) or1(x)')
    return axis


def latent_anchors(tokens):
    integer(tokens, 'tokens', 1)
    return tuple(17 * (j // 5) + 4 * (j % 5) for j in range(tokens))


def latent_warp_plan(tokens, actual_rgb_frames):
    integer(actual_rgb_frames, 'actual_rgb_frames', 1)
    anchors = latent_anchors(tokens)
    return tuple({'latent_pair': [j, j + 1], 'rgb_anchor_pair': [a, b],
                  'valid': a < actual_rgb_frames and b < actual_rgb_frames,
                  'meaning': 'encoder dependency anchors, not point-sampled RGB latents'}
                 for j, (a, b) in enumerate(zip(anchors, anchors[1:])))


def adjacent_pairs(frame_ids):
    """Only true global-time adjacency; do not bridge packed sparse-array gaps."""
    ids = tuple(frame_ids)
    if any(type(i) is not int or i < 0 or i >= 226 for i in ids):
        raise ValueError('Frame ids must be integers in0..225')
    if tuple(sorted(set(ids))) != ids:
        raise ValueError('Frame ids must be strictly increasing and unique')
    return tuple((a, b) for a, b in zip(ids, ids[1:]) if b == a + 1)


def frame_recipe(frame):
    """Released temporal assembly of one frame, prior to spatial visible crop.

    A contribution records block decoder raw frame0..27, latent slice, and
    exact crossfade weight. Weight0 is retained to describe released assembly;
    it is NOT evidence of a nonzero gradient through that contribution.
    """
    integer(frame, 'frame')
    if frame >= 226:
        raise ValueError('Expected frame0..225')

    def contribution(block, raw, numerator=1, denominator=1):
        return {'block_index': block, 'latent_start': 5 * block,
                'latent_stop': 5 * block + 7, 'raw_frame_index': raw,
                'weight': fraction_json(Fraction(numerator, denominator))}

    if frame >= 221:
        pieces = [contribution(12, 23 + frame - 221)]
        region = 'nonblend_tail'
    else:
        block, local = divmod(frame, 17)
        if block and local < 5:
            pieces = [contribution(block - 1, 23 + local, 5 - local, 5),
                      contribution(block, 3 + local, local, 5)]
            region = 'blend'
        else:
            pieces = [contribution(block, 3 + local)]
            region = 'nonblend'
    return {'rgb_frame': frame, 'region': region, 'contributions': pieces}


def temporal_regions():
    blend = tuple(i for i in range(226) if frame_recipe(i)['region'] == 'blend')
    nonblend = tuple(i for i in range(226) if i not in set(blend))
    return {'all': tuple(range(226)), 'blend': blend, 'nonblend': nonblend,
            'final_appended_tail': tuple(range(221, 226))}


def temporal_pair_groups():
    blend = set(temporal_regions()['blend'])
    pairs = adjacent_pairs(range(226))
    return {'all': pairs,
            'blend_touching': tuple(p for p in pairs if p[0] in blend or p[1] in blend),
            'nonblend_only': tuple(p for p in pairs if p[0] not in blend and p[1] not in blend),
            'blend_entry_exit': tuple(p for p in pairs if (p[0] in blend) != (p[1] in blend))}


def temporal_block_frames(block, *, scope):
    integer(block, 'block')
    if block >= 13 or scope not in ('assembled_all', 'nonblend_only'):
        raise ValueError('Require block0..12 and explicit temporal scope')
    ids = tuple(range(17 * block, 226 if block == 12 else 17 * (block + 1)))
    if scope == 'nonblend_only':
        ids = tuple(i for i in ids if frame_recipe(i)['region'] != 'blend')
    return ids


def area_weights(source_length, target_length, target_index):
    """Exact box-area integration weights, not bilinear/bicubic coordinates."""
    integer(source_length, 'source_length', 1)
    integer(target_length, 'target_length', 1)
    integer(target_index, 'target_index')
    if target_index >= target_length:
        raise ValueError('Target sample outside axis')
    left = Fraction(target_index * source_length, target_length)
    right = Fraction((target_index + 1) * source_length, target_length)
    width = right - left
    return tuple((i, (min(Fraction(i + 1), right) - max(Fraction(i), left)) / width)
                 for i in range(math.floor(left), math.ceil(right)))


def synthetic_hr_to_lr_axis_weights(axis, lr_index):
    """Proposal: edge-pad native1080 by4+4, then exact area to704x1280.

    Returned indices refer to visible HR; repeated edge-pad contributors are
    merged. This function transforms no images and assumes no production reader.
    """
    axis = checked_axis(axis)
    geometry = production67()
    result = {}
    for raw, weight in area_weights(geometry.raw_hw[axis], geometry.input_hw[axis], lr_index):
        visible = min(max(raw - (4 if axis == 0 else 0), 0), geometry.visible_hw[axis] - 1)
        result[visible] = result.get(visible, Fraction(0)) + weight
    return tuple(sorted(result.items()))


def synthetic_degradation_contract(*, color_domain):
    if color_domain not in ('srgb_code_values_float', 'linear_rgb_float'):
        raise ValueError('Require an explicit resampling color domain')
    return {'schema': 'h3-production67-synthetic-pad-area-v1-proposal',
            'status': 'unadopted_synthetic_degradation_proposal',
            'input_native_hr_hw': [1080, 1920],
            'padding': {'top': 4, 'bottom': 4, 'left': 0, 'right': 0, 'mode': 'edge_replicate'},
            'padded_hr_hw': [1088, 1920], 'lr_hw': [704, 1280],
            'resampling': 'exact separable box-area integration over half-open pixel cells',
            'color_domain': color_domain,
            'color_conversion_implemented': False, 'quantization_implemented': False,
            'lr_no_extra_crop': True, 'hr_visible_crop': [4, 1084, 0, 1920],
            'private_production_reader_equivalence': False,
            'native_hr_required': True, 'external_temporal_padding_allowed': False,
            'formal_training_allowed': False}
