"""CPU guards against changed evaluation contracts or premature accelerator work."""
import copy
import json
import sys
import unittest
from pathlib import Path

from run_tf2500_evaluation import training_complete, validate_protocol


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).resolve().parents[1] / 'config'
        parent = config / 'evaluation_parent_v1.json'
        if not parent.exists():
            parent = config / 'evaluation_v1.json'
        self.original = json.loads(parent.read_text())
        self.policy = json.loads((config / 'evaluation_tf2500_v1.json').read_text())
        self.status = {'status': 'TF_training', 'optimizer_updates': 2500, 'new_optimizer_updates': 2300}
        self.launch = {'status': 'failed', 'exit_code': -15, 'child_reaped': True}
        self.stop = {'status': 'user_requested_checkpoint_stop_complete',
                     'target_optimizer_updates': 2500, 'checkpoint_step': 2500,
                     'actual_logged_optimizer_updates': 2500, 'overshoot_updates': 0,
                     'child_reaped': True, 'parent_and_child_absent': True,
                     'raw_launcher_exit_code': -15}

    def test_new_cohort_preserves_original_contract(self):
        validate_protocol(self.policy, self.original)

    def test_reject_mutated_frozen_inputs_and_gates(self):
        for key in ('dev', 'geometry', 'shared_input', 'gate', 'metrics', 'decoder', 'selection'):
            changed = copy.deepcopy(self.policy)
            changed[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed, self.original)

    def test_reject_posthoc_checkpoint_or_execution_selection(self):
        for key, value in [('checkpoints', [2000, 2500]),
                           ('execution_checkpoint_order', [2500]),
                           ('arms', ['strict_streaming_step2500']),
                           ('step0_identity_guaranteed', True), ('test_unlocked', True)]:
            changed = copy.deepcopy(self.policy)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed, self.original)

    def test_terminal_reaped_success_can_proceed(self):
        self.assertTrue(training_complete(self.status, self.launch, {'ready': True}, self.stop))

    def test_live_or_unreaped_predecessor_refused(self):
        self.assertFalse(training_complete(self.status, self.launch, {'ready': False}, self.stop))
        for key, value in [('child_reaped', False), ('exit_code', 1), ('status', 'running')]:
            launch = dict(self.launch, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(training_complete(self.status, launch, {'ready': True}, self.stop))

    def test_partial_or_wrong_update_count_refused(self):
        for key, value in [('optimizer_updates', 2000), ('new_optimizer_updates', 2500),
                           ('optimizer_updates', 2501)]:
            status = dict(self.status, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(training_complete(status, self.launch, {'ready': True}, self.stop))

    def test_missing_or_inconsistent_intentional_stop_refused(self):
        self.assertFalse(training_complete(self.status, self.launch, {'ready': True}, None))
        for key, value in [('target_optimizer_updates', 3000), ('checkpoint_step', 2000),
                           ('actual_logged_optimizer_updates', 2499), ('overshoot_updates', 1),
                           ('child_reaped', False), ('parent_and_child_absent', False),
                           ('raw_launcher_exit_code', 1), ('status', 'waiting')]:
            stop = dict(self.stop, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(training_complete(self.status, self.launch, {'ready': True}, stop))

    def test_overshoot_must_be_reported_honestly(self):
        status = dict(self.status, optimizer_updates=2501, new_optimizer_updates=2301)
        stop = dict(self.stop, actual_logged_optimizer_updates=2501, overshoot_updates=1)
        self.assertTrue(training_complete(status, self.launch, {'ready': True}, stop))
        self.assertFalse(training_complete(self.status, self.launch, {'ready': True}, stop))

    def test_admission_import_has_no_accelerator_runtime(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('torch_npu', sys.modules)


if __name__ == '__main__':
    unittest.main()
