import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from image_builder import builder as builder_module
from image_builder import verifier as verifier_module
from image_builder.contracts import INPUT_ROLES, RESTORED_FILESYSTEM_SPECS
from image_builder.errors import BuilderError
from image_builder.geometry import DISK_BYTES, PARTITION_BY_NUMBER, SECTOR_SIZE
from image_builder.update_inputs import (
    PARTCLONE_MAGIC,
    UPDATE_PATHS,
    _prepare_update_inputs,
    signed_crc32,
)


PACKAGE = "TEST_UPDATE"


def _catalog(payloads: dict[str, bytes]) -> bytes:
    rows = [f"+|1|DHPE.TEST|HM|{PACKAGE}|0|1"]
    for record_id, role in enumerate(INPUT_ROLES, start=10):
        relative = UPDATE_PATHS[role]
        payload = payloads[role]
        crc = signed_crc32(zlib.crc32(payload) & 0xFFFFFFFF)
        directory = "\\".join((PACKAGE, *relative.parent.parts))
        rows.append(
            f"{directory}|{relative.name}|{record_id}|{crc}|"
            f"{len(payload)}|1"
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def _write_update(
    update: Path,
    payloads: dict[str, bytes],
) -> Path:
    for role, relative in UPDATE_PATHS.items():
        target = update / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[role])
    (update / f"{PACKAGE}.ver").write_bytes(_catalog(payloads))
    update_key = update.parent / "genesis.update-key"
    update_key.write_bytes(bytes(16))
    return update_key


class VerificationTests(unittest.TestCase):
    def test_update_package_uses_shipped_integrity_values(self) -> None:
        payloads = {
            "boot": b"encrypted boot container",
            "root": b"encrypted root container",
            "vr": PARTCLONE_MAGIC + b" voice image",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            update = root / "update"
            work = root / "work"
            update.mkdir()
            work.mkdir()
            update_key = _write_update(update, payloads)

            def run_tool(command, *, description, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                if description.startswith("decrypt"):
                    self.assertIn("--key", command)
                    self.assertIn("--expect-partclone", command)
                    output.write_bytes(PARTCLONE_MAGIC)
                    return
                self.assertIn("--restore_raw_file", command)
                role = description.split()[1]
                with output.open("wb") as stream:
                    stream.truncate(
                        int(RESTORED_FILESYSTEM_SPECS[role]["bytes"])
                    )

            with (
                mock.patch(
                    "image_builder.update_inputs.UPDATE_KEY_PATH",
                    update_key,
                ),
                mock.patch(
                    "image_builder.update_inputs.require_host_tool",
                    side_effect=lambda name, **_kwargs: f"/usr/bin/{name}",
                ),
                mock.patch(
                    "image_builder.update_inputs.compile_decoder",
                    return_value=Path("/mock/decrypt_lg_container"),
                ),
                mock.patch(
                    "image_builder.update_inputs._run_checked",
                    side_effect=run_tool,
                ) as run,
            ):
                with _prepare_update_inputs(
                    update,
                    work_directory=work,
                ) as prepared:
                    checks = prepared.update_verification["checks_passed"]
                    self.assertEqual(
                        checks,
                        {
                            "catalog_byte_size_and_crc32": list(INPUT_ROLES),
                            "lg_trailer_and_encrypted_payload_sha256": [
                                "boot",
                                "root",
                            ],
                            "partclone_record_crc": list(INPUT_ROLES),
                        },
                    )
                    self.assertTrue(prepared.boot_path.exists())
                    self.assertTrue(prepared.root_path.exists())
                    self.assertTrue(prepared.vr_path.exists())
                self.assertEqual(run.call_count, 5)

            update_key.write_bytes(bytes(15))
            with (
                mock.patch(
                    "image_builder.update_inputs.UPDATE_KEY_PATH",
                    update_key,
                ),
                mock.patch(
                    "image_builder.update_inputs.require_host_tool"
                ) as host_tool,
                self.assertRaisesRegex(
                    BuilderError,
                    "update key is 15 bytes",
                ),
            ):
                with _prepare_update_inputs(
                    update,
                    work_directory=work,
                ):
                    self.fail("invalid update key was accepted")
            host_tool.assert_not_called()
            update_key.write_bytes(bytes(16))

            boot_path = update / UPDATE_PATHS["boot"]
            corrupted = bytearray(boot_path.read_bytes())
            corrupted[0] ^= 0xFF
            boot_path.write_bytes(corrupted)
            with (
                mock.patch(
                    "image_builder.update_inputs.UPDATE_KEY_PATH",
                    update_key,
                ),
                mock.patch(
                    "image_builder.update_inputs.require_host_tool"
                ) as host_tool,
                self.assertRaisesRegex(
                    BuilderError,
                    "catalog CRC-32 mismatch",
                ),
            ):
                with _prepare_update_inputs(
                    update,
                    work_directory=work,
                ):
                    self.fail("corrupted update image was accepted")
            host_tool.assert_not_called()

    def test_invalid_source_partition_stops_before_customization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {}
            for role in INPUT_ROLES:
                path = root / f"{role}.filesystem"
                path.write_bytes(b"x")
                inputs[role] = path
            output = root / "headunit.img"

            with (
                mock.patch.object(
                    builder_module.os,
                    "geteuid",
                    return_value=1000,
                ),
                mock.patch.object(
                    builder_module,
                    "_preflight_output_directory",
                ),
                mock.patch.object(
                    builder_module,
                    "stage_jukebox_fix",
                ) as customize,
                mock.patch.object(
                    builder_module,
                    "publish_no_replace",
                ) as publish,
                mock.patch.object(
                    builder_module.tempfile,
                    "mkdtemp",
                ) as make_workdir,
                self.assertRaisesRegex(
                    BuilderError,
                    "boot filesystem is 1 bytes",
                ),
            ):
                builder_module._assemble_prepared_image(
                    boot_path=inputs["boot"],
                    root_path=inputs["root"],
                    vr_path=inputs["vr"],
                    output_path=output,
                )

            customize.assert_not_called()
            make_workdir.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(output.exists())

    def test_assembled_partition_hashes_gate_publication(self) -> None:
        expected_hashes = {
            "boot": "1" * 64,
            "root": "2" * 64,
            "vr": "3" * 64,
        }
        number_by_offset = {
            partition.start * SECTOR_SIZE: number
            for number, partition in PARTITION_BY_NUMBER.items()
        }

        def fake_superblock(_fd, offset=0):
            number = number_by_offset[offset]
            role = verifier_module.IMMUTABLE_PARTITIONS.get(number)
            if role is not None:
                filesystem_bytes = int(
                    RESTORED_FILESYSTEM_SPECS[role]["bytes"]
                )
            else:
                spec = verifier_module.fresh_ext_spec(number)
                filesystem_bytes = int(spec["block_count"]) * int(
                    spec["block_size"]
                )
            return SimpleNamespace(
                filesystem_bytes=filesystem_bytes,
                as_dict=lambda: {"filesystem_bytes": filesystem_bytes},
            )

        def fake_partition_hash(_fd, offset, _size):
            number = number_by_offset[offset]
            role = verifier_module.IMMUTABLE_PARTITIONS[number]
            return expected_hashes[role]

        with (
            mock.patch.object(
                verifier_module,
                "verify_table",
                return_value={"disk_id": 1},
            ),
            mock.patch.object(
                verifier_module,
                "parse_superblock",
                side_effect=fake_superblock,
            ),
            mock.patch.object(verifier_module, "validate_superblock"),
            mock.patch.object(
                verifier_module,
                "_require_superblock_values",
            ),
            mock.patch.object(
                verifier_module,
                "_parse_swap",
                return_value={"partition": 3},
            ),
            mock.patch.object(
                verifier_module,
                "sha256_range_fd",
                side_effect=fake_partition_hash,
            ),
        ):
            result = verifier_module._verify_assembled_image(
                -1,
                DISK_BYTES,
                expected_payload_sha256=expected_hashes,
            )
            self.assertEqual(result["payload_sha256"], expected_hashes)

            with self.assertRaisesRegex(
                BuilderError,
                "assembled root filesystem differs",
            ):
                verifier_module._verify_assembled_image(
                    -1,
                    DISK_BYTES,
                    expected_payload_sha256={
                        **expected_hashes,
                        "root": "f" * 64,
                    },
                )

        validation = {
            "schema": "genesis-partition-validation/v1",
            "stage": "source",
            "valid": True,
            "partitions": {
                role: {
                    "bytes": int(RESTORED_FILESYSTEM_SPECS[role]["bytes"]),
                    "sha256": digest,
                    "filesystem": {},
                }
                for role, digest in expected_hashes.items()
            },
        }
        sources = {
            role: SimpleNamespace(
                fd=-1,
                size=int(RESTORED_FILESYSTEM_SPECS[role]["bytes"]),
                assert_unchanged=lambda: None,
            )
            for role in INPUT_ROLES
        }
        source_stack = mock.Mock()
        source_stack.enter_context.return_value = SimpleNamespace(
            fd=-1,
            size=int(RESTORED_FILESYSTEM_SPECS["root"]["bytes"]),
            assert_unchanged=lambda: None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mismatch.img"

            def create_empty(path, *_args):
                path.write_bytes(b"")
                return SimpleNamespace()

            staged_root = mock.MagicMock()
            staged_root.fd = -1
            staged_root.size = 0
            staged_root.assert_unchanged = mock.Mock()
            staged_root.__enter__.return_value = staged_root
            jukebox_report = {
                "schema": "genesis-jukebox-storage-fix/v1",
            }

            with (
                mock.patch.object(
                    builder_module.os,
                    "geteuid",
                    return_value=1000,
                ),
                mock.patch.object(builder_module, "DISK_BYTES", 4096),
                mock.patch.object(
                    builder_module,
                    "_preflight_output_directory",
                ),
                mock.patch.object(
                    builder_module,
                    "_validate_prepared_filesystems",
                    return_value=(validation, source_stack, sources),
                ),
                mock.patch.object(
                    builder_module,
                    "extract_syslinux_mbr",
                    return_value=bytes(440),
                ),
                mock.patch.object(builder_module, "write_table"),
                mock.patch.object(
                    builder_module,
                    "create_swap",
                    side_effect=create_empty,
                ),
                mock.patch.object(
                    builder_module,
                    "create_legacy_ext",
                    side_effect=create_empty,
                ),
                mock.patch.object(
                    builder_module,
                    "populate_p7",
                    return_value=builder_module.P7_SKELETON_SHA256,
                ),
                mock.patch.object(builder_module, "run_e2fsck_path"),
                mock.patch.object(
                    builder_module,
                    "stage_jukebox_fix",
                    return_value=(staged_root, jukebox_report),
                ) as customize,
                mock.patch.object(
                    builder_module,
                    "open_regular_read",
                    return_value=staged_root,
                ),
                mock.patch.object(
                    builder_module,
                    "sha256_fd",
                    return_value=expected_hashes["root"],
                ),
                mock.patch.object(builder_module, "copy_sparse_to_offsets"),
                mock.patch.object(
                    builder_module,
                    "_verify_assembled_image",
                    side_effect=BuilderError(
                        "assembled root filesystem differs from the "
                        "verified build payload"
                    ),
                ) as verify,
                mock.patch.object(
                    builder_module,
                    "publish_no_replace",
                ) as publish,
                self.assertRaisesRegex(
                    BuilderError,
                    "assembled root filesystem differs",
                ),
            ):
                builder_module._assemble_prepared_image(
                    boot_path="boot",
                    root_path="root",
                    vr_path="vr",
                    output_path=output,
                )

            customize.assert_called_once()
            verify.assert_called_once()
            publish.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
