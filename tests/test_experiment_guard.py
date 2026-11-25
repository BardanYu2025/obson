"""Exercise release immutability and isolation using disposable Git repositories."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('guard', Path(__file__).resolve().parents[1] / 'tools/experiment_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ExperimentGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.folder = self.make_experiment('20261125_fixture_v1')
        guard.seal(self.root, self.folder.name)
        self.commit()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')

    def make_experiment(self, name):
        folder = self.root / 'experiments' / name
        (folder / 'src').mkdir(parents=True)
        (folder / 'reports').mkdir()
        (folder / 'README.md').write_text('Isolated CPU fixture\n')
        (folder / 'requirements.lock').write_text('# Standard library only\n')
        (folder / 'src/audit.py').write_text('import json\n')
        (folder / 'reports/original.json').write_text('{}\n')
        meta = {'schema':'babel.experiment.v1', 'id':name, 'status':'released',
                'kind':'data_audit', 'entrypoints':{'audit':'src/audit.py'},
                'source_files':['src/audit.py'], 'ready_for_execution':True}
        (folder / 'experiment.json').write_text(json.dumps(meta))
        return folder

    def test_unchanged_release_and_new_report_pass(self):
        (self.folder / 'reports/new.json').write_text('{"new": true}\n')
        self.assertEqual(guard.check(self.root, 'HEAD'), 1)

    def test_mutated_code_is_rejected(self):
        (self.folder / 'src/audit.py').write_text('print("different")\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source'):
            guard.check(self.root, 'HEAD')

    def test_rewriting_hashes_does_not_evade_baseline(self):
        file = self.folder / 'src/audit.py'
        file.write_text('print("different")\n')
        seal = self.folder / 'release.json'
        record = json.loads(seal.read_text())
        record['files']['src/audit.py'] = hashlib.sha256(file.read_bytes()).hexdigest()
        seal.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Published file overwritten'):
            guard.check(self.root, 'HEAD')

    def test_deleting_entire_frozen_experiment_is_rejected(self):
        shutil.rmtree(self.folder)
        with self.assertRaisesRegex(ValueError, 'Frozen file deleted'):
            guard.check(self.root, 'HEAD')

    def test_new_revision_can_coexist_with_frozen_original(self):
        other = self.make_experiment('20261125_fixture_v2')
        (other / 'src/audit.py').write_text('import math\n')
        guard.seal(self.root, other.name)
        self.assertEqual(guard.check(self.root, 'HEAD'), 2)

    def test_old_package_import_cannot_be_sealed(self):
        other = self.make_experiment('20261125_fixture_v2')
        (other / 'src/audit.py').write_text('from obson.babel import data\n')
        with self.assertRaisesRegex(ValueError, 'legacy import'):
            guard.seal(self.root, other.name)

    def test_sibling_experiment_import_cannot_be_sealed(self):
        other = self.make_experiment('20261125_fixture_v2')
        (other / 'src/audit.py').write_text('from experiments import something\n')
        with self.assertRaisesRegex(ValueError, 'Cross-experiment'):
            guard.seal(self.root, other.name)

    def test_external_symlink_is_rejected(self):
        other = self.make_experiment('20261125_fixture_v2')
        (other / 'src/shared.py').symlink_to(self.folder / 'src/audit.py')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            guard.seal(self.root, other.name)

    def test_new_source_file_cannot_be_added_to_release(self):
        (self.folder / 'src/new.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source'):
            guard.check(self.root, 'HEAD')

    def test_report_added_after_release_becomes_immutable_at_commit(self):
        report = self.folder / 'reports/new.json'
        report.write_text('{"loss": 5}\n')
        self.commit()
        report.write_text('{"loss": 1}\n')
        with self.assertRaisesRegex(ValueError, 'Published file overwritten'):
            guard.check(self.root, 'HEAD')

    def test_cannot_delete_and_reseal_published_id(self):
        (self.folder / 'release.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Previously published'):
            guard.seal(self.root, self.folder.name)

    def test_entrypoint_cannot_escape_directory(self):
        other = self.make_experiment('20261125_fixture_v2')
        meta = json.loads((other / 'experiment.json').read_text())
        meta['entrypoints']['audit'] = '../20261125_fixture_v1/src/audit.py'
        (other / 'experiment.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'Unsafe path'):
            guard.seal(self.root, other.name)


if __name__ == '__main__':
    unittest.main()
