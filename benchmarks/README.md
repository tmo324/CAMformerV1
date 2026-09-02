# Benchmark Records

This directory preserves supplementary GPU profiling records collected during CAMformer evaluation.

For each platform, `*_raw_profile.txt` is the raw text record and `*_profile_summary.xlsx` is its compact summary:

- NVIDIA A100
- NVIDIA L40
- NVIDIA Titan Xp

These records are archival inputs for audit and follow-on analysis. The default `make paper` workflow does not execute GPU profiling and does not consume these files. Hardware, driver, workload, and software differences can change new measurements.
