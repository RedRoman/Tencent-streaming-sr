"""CPU orchestration checks with a stateful numerical fake, not model evidence."""
import json
import sys
from types import SimpleNamespace
import torch
from diagnose_tf2500_history import diagnose_sample, require_ready


class FakeRuntime:
    device = torch.device('cpu')
    def __init__(self):
        self.cache = None
        self.next_frame = 0
        self.history = {}
        self.commits = []
    def start_stream(self, condition):
        self.history = {}
        self.next_frame = 0
        self.cache = []
    def values(self, x, cond, sigma, pos, history):
        allowed = {k: v for k, v in history.items() if k == 0 or k >= pos - 1}
        past = sum(allowed.values())
        value = .15*x + .05*cond + .02*past + .01*pos + .03*sigma
        hidden = float(x.mean()) + .1*past + .02*float(cond.mean())
        return value, hidden
    def stream_velocity(self, x, condition, sigma, commit=False):
        value, hidden = self.values(x, condition, sigma, self.next_frame, self.history)
        if commit:
            assert sigma == 0
            self.commits.append(self.next_frame)
            self.history[self.next_frame] = hidden
            self.next_frame += 1
            self.history = {k: v for k, v in self.history.items() if k == 0 or k >= self.next_frame - 1}
            keys = sorted(self.history)
            self.cache = [SimpleNamespace(k=torch.tensor([self.history[k] for k in keys]),
                                          v=torch.ones(len(keys)), frames=tuple(keys))]
        return value
    def offline(self, x, condition, sample, frames, sigma):
        history = {}
        for index, pos in enumerate(frames):
            value, hidden = self.values(x[:, index:index+1], condition[:, index:index+1],
                                        sigma if index == len(frames)-1 else 0, pos, history)
            history[pos] = hidden
        return value


def noise(shape, sid, pos, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed + pos))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    plan = {'positions': [0, 3, 11], 'sigmas': [.909375, 6/7, 2/3, 0.], 'seed': 77}
    sample = {'sample_id': 'synthetic_control', 'reference': torch.randn(2, 12, 2, 2),
              'upscaled': torch.randn(2, 12, 2, 2)}
    rt = FakeRuntime()
    rt.start_stream(None)
    values = []
    for pos in range(12):
        cond = sample['upscaled'][:, pos:pos+1]
        x = (1-plan['sigmas'][0])*cond + plan['sigmas'][0]*noise(cond.shape, '', pos, plan['seed'])
        for sigma, following in zip(plan['sigmas'], plan['sigmas'][1:]):
            x = x + (sigma-following)*rt.stream_velocity(x, cond, sigma)
        rt.stream_velocity(x, cond, 0, commit=True)
        values.append(x)
    rt.commits = []
    report = diagnose_sample(rt, sample, None, torch.cat(values, 1), plan, noise, lambda *a, **kw: None)
    assert len(report['rows']) == 36 and len(report['history_differences']) == 24
    assert report['model_forwards_including_static_and_history_writes'] == 81
    assert rt.commits == list(range(12))*2
    assert report['four_latent_full_prefix_oracle_parity'][0]['max_abs'] == 0
    assert all(v['saved_self_history_replay']['max_abs'] == 0 for v in report['rows'] if 'saved_self_history_replay' in v)
    assert all(v['difference']['max_abs'] == 0 for v in report['history_differences'] if v['position'] == 0)
    assert any(v['difference']['max_abs'] > 0 for v in report['history_differences'] if v['position'] == 3 and v['mode'] == 'gt_truncated_offline')
    for report_bad in [{'ready_for_NPU': False}]:
        try:
            require_ready(report_bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Active predecessor was allowed')
    result = {'status': 'passed', 'scope': 'stateful numerical fake on CPU; no real-model quality claim',
              'matched_rows': 36, 'history_comparisons': 24, 'expected_forwards': 81,
              'absolute_commit_order': 'two sequential0..11 histories; probes never commit',
              'four_latent_oracle_max_abs': 0, 'self_history_replay_max_abs': 0,
              'no_history_control_max_abs': 0, 'truncated_context_difference_detected': True,
              'active_predecessor_refused': True, 'new_optimizer_updates': 0,
              'accelerator_imported': 'torch_npu' in sys.modules, 'new_accelerator_forwards': 0}
    assert not result['accelerator_imported']
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
