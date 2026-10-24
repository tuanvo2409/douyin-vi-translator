from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


WORKER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WORKER_ROOT.parent
FIXTURE_ROOT = WORKER_ROOT / "tests" / "fixtures" / "p1c_engine_contract"
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

import dubvi_engine_contract as contract


class TranslatorEngineContractTests(unittest.TestCase):
    def _fixture(self, name: str) -> dict[str, object]:
        return contract.parse_json_document((FIXTURE_ROOT / name).read_bytes())

    def test_vendored_module_and_fixture_manifest_are_pinned(self) -> None:
        manifest = self._fixture("contract-manifest.json")
        module_hash = hashlib.sha256((WORKER_ROOT / "dubvi_engine_contract.py").read_bytes()).hexdigest()
        self.assertEqual(contract.CONTRACT_IMPLEMENTATION_VERSION, manifest["contract_implementation_version"])
        self.assertEqual(module_hash, manifest["module_sha256"])
        listed = manifest["fixtures"]
        self.assertEqual(
            set(listed),
            {path.name for path in FIXTURE_ROOT.glob("*.json") if path.name != "contract-manifest.json"},
        )
        for name, expected in listed.items():
            self.assertEqual(expected, hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest())

    def test_complete_shared_fixture_set_has_identical_behavior(self) -> None:
        for path in sorted(FIXTURE_ROOT.glob("*.valid.json")):
            with self.subTest(path=path.name):
                contract.validate_document(contract.parse_json_document(path.read_bytes()))
        for path in sorted(FIXTURE_ROOT.glob("*.invalid.json")):
            with self.subTest(path=path.name):
                with self.assertRaises(contract.ContractValidationError):
                    contract.validate_document(contract.parse_json_document(path.read_bytes()))

    def test_v2_handoff_and_translator_attempt_retries_are_valid(self) -> None:
        handoff = self._fixture("reup_to_translator_handoff_v2.valid.json")
        self.assertNotIn("translator_job_id", handoff)
        contract.validate_document(handoff)
        contract.validate_document(self._fixture("cp_to_translator_job.valid.json"))
        retry = self._fixture("cp_to_translator_job.retry.valid.json")
        self.assertEqual(2, retry["attempt_number"])
        self.assertNotEqual(retry["translator_job_id"], "77777777-7777-4777-8777-777777777777")
        contract.validate_document(retry)

    def test_translator_status_fixtures_are_valid(self) -> None:
        for name in (
            "translator.accepted.valid.json",
            "translator.review_required.valid.json",
            "translator.rejected.valid.json",
            "translator.failed.valid.json",
            "translator.succeeded.valid.json",
        ):
            with self.subTest(name=name):
                contract.validate_document(self._fixture(name))

    def test_invalid_reference_fingerprint_and_correlation_fail(self) -> None:
        for name in (
            "cp_to_translator_job.absolute_ref.invalid.json",
            "cp_to_reup_job.bad_fingerprint.invalid.json",
            "cp_to_reup_job.bad_correlation.invalid.json",
            "handoff_v2.translator_job_id.invalid.json",
        ):
            with self.subTest(name=name):
                with self.assertRaises(contract.ContractValidationError):
                    contract.validate_document(self._fixture(name))


if __name__ == "__main__":
    unittest.main()
