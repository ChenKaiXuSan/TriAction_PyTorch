Archived trainer implementations that are no longer used by the current
training entrypoint.

The active training path is:

- `project.main`
- `project.trainer.single_selector`
- `project.trainer.multi_selector`
- `project.trainer.single.train_single_modality`
- `project.trainer.multi.early.train_early_fusion`
- `project.trainer.multi.mid.train_multi_ts_cva`
- `project.trainer.multi.late.train_late_fusion`

Files under this directory are retained as historical references only. They are
not imported by the current experiment matrix or selector-based training flow.
