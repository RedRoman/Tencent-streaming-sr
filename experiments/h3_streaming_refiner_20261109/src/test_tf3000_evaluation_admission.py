"""CPU guards against changed evaluation contracts or premature accelerator work."""
import copy
import json
import sys
import unittest
from pathlib import Path

from run_tf3000_evaluation import training_complete, validate_protocol


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).resolve().parents[1] / 'config'
        parent = config / 'evaluation_parent_v1.json'
        if not parent.exists():
            parent = config / 'evaluation_v1.json'
        self.original = json.loads(parent.read_text())
        self.policy = json.loads((config / 'evaluation_tf3000_v1.json').read_text())
        self.status = {'status': 'TF3000_complete_evaluation_pending',
                       'optimizer_updates': 3000, 'new_optimizer_updates': 2800}
        self.launch = {'status': 'complete', 'exit_code': 0, 'child_reaped': True}

    def test_new_cohort_preserves_original_contract(self):
        validate_protocol(self.policy, self.original)

    def test_reject_mutated_frozen_inputs_and_gates(self):
        for key in ('dev', 'geometry', 'shared_input', 'gate', 'metrics', 'decoder', 'selection'):
            changed = copy.deepcopy(self.policy)
            changed[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed, self.original)

    def test_reject_posthoc_checkpoint_or_execution_selection(self):
        for key, value in [('checkpoints', [2000, 3000]),
                           ('execution_checkpoint_order', [3000]),
                           ('arms', ['strict_streaming_step3000']),
                           ('step0_identity_guaranteed', True), ('test_unlocked', True)]:
            changed = copy.deepcopy(self.policy)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed, self.original)

    def test_terminal_reaped_success_can_proceed(self):
        self.assertTrue(training_complete(self.status, self.launch, {'ready': True}))

    def test_live_or_unreaped_predecessor_refused(self):
        self.assertFalse(training_complete(self.status, self.launch, {'ready': False}))
        for key, value in [('child_reaped', False), ('exit_code', 1), ('status', 'running')]:
            launch = dict(self.launch, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(training_complete(self.status, launch, {'ready': True}))

    def test_partial_or_wrong_update_count_refused(self):
        for key, value in [('optimizer_updates', 2000), ('new_optimizer_updates', 3000),
                           ('status', 'TF_training')]:
            status = dict(self.status, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(training_complete(status, self.launch, {'ready': True}))

    def test_admission_import_has_no_accelerator_runtime(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('torch_npu', sys.modules)


if __name__ == '__main__':
    unittest.main()
