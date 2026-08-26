# Safetensors checkpoint manifest

`MODELS.md` maps paper model names to these files and their inference folders.
Every file below contains only tensors and has no safetensors metadata payload.

| Release-relative path | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/ablations/grt_no_doppler/grt_no_doppler.safetensors` | 119293056 | `3c1ce6390ca02ef51b3ed4bda10a24a88d48a6e3e6d46b6ed97862c8ee935569` |
| `checkpoints/ablations/grt_refine/control.safetensors` | 1456999608 | `1e959efd44348813ff16bd407409e3aee59a995a9792dd6a8dbbf5c80ab9e219` |
| `checkpoints/ablations/grt_refine/diffusion.safetensors` | 3463772560 | `9839e251d1aa77ee7e01223be64b41adfcebc00c8f419e49ddf3f74e0f600bbd` |
| `checkpoints/ablations/grt_refine/grt.safetensors` | 119293056 | `136286fbf25f5d44fb60babb5b72cf8540c8ea0ac8b6da5b1d70bc4f615d269f` |
| `checkpoints/ablations/ours_full_no_3d/control.safetensors` | 1456999608 | `b6bdcd0a80295afa69741ec48199bc8383ba28808696b9de6b6580c58b8a8266` |
| `checkpoints/ablations/ours_full_no_3d/diffusion.safetensors` | 3463772560 | `41cae148986fc3c750ee1d7588d67b3ef13c76c3b959f223517ee1907355ec67` |
| `checkpoints/ablations/ours_full_no_3d/radar.safetensors` | 28153012 | `c4125ae3670f646d1e60639f97b27d4e6b6c2f995cdb818e90c9d36198727d34` |
| `checkpoints/ablations/ours_radar_no_doppler/ours_radar_no_doppler.safetensors` | 28153012 | `87558d1c95447cfcac482edc88387a58ef9def78a09511a75c009e23ae3adb55` |
| `checkpoints/ablations/ours_radar_no_grad/ours_radar_no_grad.safetensors` | 28153012 | `a00af78eb8ff5e2332d1aac042ab463f046cd09f18f107edea9d5bc8effa4cf9` |
| `checkpoints/baselines/cafnet/cafnet.safetensors` | 249249292 | `025e0da0c545aff30934158bdf6e8ae628ca4e6305892585d870ffdacf102b14` |
| `checkpoints/baselines/cafnet_no_smoke/cafnet_no_smoke.safetensors` | 249249292 | `fe055af6fd4b0f95ca7b6c5574c8c3c63fcf4bad3549a9d3ade92910e23a844d` |
| `checkpoints/baselines/da3/da3metric-large.safetensors` | 1336734448 | `bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776` |
| `checkpoints/baselines/grt/grt.safetensors` | 119293056 | `136286fbf25f5d44fb60babb5b72cf8540c8ea0ac8b6da5b1d70bc4f615d269f` |
| `checkpoints/baselines/grt_image/grt_image.safetensors` | 165108504 | `862091505388d24ef3c8b7a2877dc869c3d9cb78c268ed7b68cf7b56356150b3` |
| `checkpoints/baselines/radarcam-depth/radarcam-depth_rcnet.safetensors` | 24022648 | `9f7ff7d26d0c2c3183e082fe563210d768ff54fdb0506ca7405468805df39679` |
| `checkpoints/baselines/radarcam-depth/radarcam-depth_sml.safetensors` | 85629216 | `d77f26766e37bd8153d8208aa2e771205436582e37caf8a989a28adf43bbd2b5` |
| `checkpoints/grade/control.safetensors` | 1456999608 | `6bc3bf8e520a1ba5ac672cc587a17c50398cfa4a49c0b17b0b4fc634bab53e3e` |
| `checkpoints/grade/diffusion.safetensors` | 3463772560 | `5c9ff00df7e9ce75217f0ca900503c3a6e631ad902966a7bb05bae0e4017c5c8` |
| `checkpoints/grade/radar.safetensors` | 28153012 | `c4125ae3670f646d1e60639f97b27d4e6b6c2f995cdb818e90c9d36198727d34` |

The matching hashes for baseline GRT and retrained GRT, and for GRADE radar and
the no-3D radar checkpoint, are intentional: those pairs contain identical
model tensors after training-state removal.
