import importlib
import unittest


class EntrypointImportTests(unittest.TestCase):
    def test_training_entrypoints_import_without_circular_dependencies(self):
        for module_name in (
            "project.main",
            "project.trainer.single_selector",
            "project.trainer.multi_selector",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
