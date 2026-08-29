# Action items

## Before the August 28, 2026 AE deadline

1. Rotate the exposed W&B credential and verify that no credentials remain in
   source, history, archived logs, or the final artifact.
2. From the server, retrieve the exact data/checkpoint list in
   `SERVER_DOWNLOADS.md`.
3. For each paper model/ablation, make a provenance row: paper label, command,
   config SHA-256, checkpoint SHA-256, output location, and selected metric.
4. Decide data redistribution eligibility (consent/IRB/license and image/
   location privacy).  If data cannot be public, set up anonymous remote
   reviewer access and document the restriction.
5. Produce sanitized, release-relative configs.  Turn W&B off or set it to
   offline; never place API keys in YAML.
   Also scrub the remaining 14 Python files with personal absolute-path
   literals, listed in `ANONYMITY_AND_SECURITY.md`.
6. Build and test a CUDA container on the server.  Record GPU model, driver,
   CUDA, OS, Python, runtime, disk/RAM, and exact commands.
7. Execute the evaluation-only workflow from a clean environment.  Save the
   stdout tables and compare them with the camera-ready numbers. The packaged
   run matches all 20 checked paper rasters: 19 byte-for-byte and one
   (`revision/graceful_degradation_3.png`) pixel-for-pixel with only PNG
   metadata differing.
8. Execute one end-to-end representative inference-to-evaluation workflow from
   released data/checkpoint, then document a reduced scope if full evaluation
   is too expensive.
9. Complete the <=3-page artifact description, including claimed key results,
   requirements, expected duration, and troubleshooting.
   Resolve the current Figure 10 exclusion before claiming that all main
   results are reproduced.
10. Submit only the anonymous review archive through HotCRP; do not expose a
    named public GitHub/Zenodo link during review.

## After AE acceptance

1. Publish a frozen public source release and attach/archive it in Zenodo to
   mint a DOI.
2. Publish the approved data and canonical weights (or stable public records)
   with checksums and licenses; link them from the DOI landing page.
3. Add citation metadata, a project license, and third-party notices; verify
   that bundled baseline code permits redistribution.
4. Tag the final version, preserve the validation log, and update every link
   in the README and artifact description.
