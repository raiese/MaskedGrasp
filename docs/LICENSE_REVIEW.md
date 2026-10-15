# License review — author decision pending

Reviewed against local notices and official sources on 2026-10-15. No new blanket repository license has been applied. This is a preparation review, not a determination that all rights have been cleared.

| Component | Official evidence / observed license | Packaging decision |
|---|---|---|
| FreeGrasp-derived root scripts, utils, LangSAM adapters | [Code repository](https://github.com/tev-fbk/FreeGrasp_code), [LICENCE](https://github.com/tev-fbk/FreeGrasp_code/blob/main/LICENCE): CC BY-NC 4.0 | Preserve original notice and attribution; derivatives retain upstream conditions |
| ThinkGrasp-derived simulation code | [MIT license](https://github.com/H-Freax/ThinkGrasp/blob/main/LICENSE) | Preserve MIT notice; this does not override nested dependencies |
| FreeGraspData | [Dataset card](https://huggingface.co/datasets/FBK-TeV/FreeGraspData): CC BY-NC-SA 3.0 | Official link only; no annotations/images bundled |
| MetaGraspNetV2 | [Official project](https://github.com/maximiliangilles/MetaGraspNet): CC BY-NC-SA 3.0 for dataset | Official links only; confirm asset/version provenance |
| Molmo-7B-D-0924 | [Official card](https://huggingface.co/allenai/Molmo-7B-D-0924): Apache-2.0 | Model snapshot excluded |
| Qwen2.5-VL-7B-Instruct | [Official card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): Apache-2.0 | Model snapshot excluded; exact historical revision unknown |
| GroundingDINO | [Official license](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE): Apache-2.0 for code | External dependency and weight links; not vendored |
| Segment Anything | [Official license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE): Apache-2.0 | External dependency and weight links; not vendored |
| GraspNet API | [Official license](https://github.com/graspnet/graspnetAPI/blob/master/LICENSE): MIT | Install externally; distinct from graspnet-baseline |
| graspnet-baseline | [Official license](https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE), also present locally: noncommercial research agreement with redistribution/sublicensing restrictions | Model implementation kept outside candidate; obtain rights/authorized source before redistribution |
| FGC-GraspNet and local FGC implementation | [Official repository](https://github.com/luyh20/FGC-GraspNet) has no clear top-level LICENSE in inspected tree; nested files cite other terms | Model sources retained privately for review; no assertion that FreeGrasp's license clears all nested code |
| LangSAM upstream | [Official repository](https://github.com/luca-medeiros/lang-segment-anything): Apache-2.0; current upstream uses SAM 2.1, unlike these legacy SAM adapters | Retain attribution; verify historical adapter revision before assigning blanket terms |
| PyBullet robot/object URDFs, meshes, unseen object collection | Mixed local asset notices; full per-asset provenance unresolved | No asset files included; scene references only |
| BERT base uncased (optional GroundingDINO text encoder) | [Official model card](https://huggingface.co/google-bert/bert-base-uncased): Apache-2.0 | Download externally and check the chosen snapshot's terms |

The FreeGrasp and MetaGraspNet READMEs additionally state academic-use restrictions including military-use exclusion. Preserve the upstream notices and clarify any discrepancy between README wording and the linked standard license; do not invent a new combined permission grant.

## Suggested decision

For this existing derivative code, preserve FreeGrasp's CC BY-NC 4.0 attribution and conditions, retain ThinkGrasp's MIT notices for its portions, and document each external component separately. This is a **research/source-available release**, not an unrestricted OSI open-source release: the [Open Source Definition](https://opensource.org/osd) does not allow discrimination against fields of endeavor.

If a permissive open-source release is desired, Apache-2.0 is a reasonable candidate **only for clearly owned, separable new code after author/institution approval**. Do not place the entire current repository under MIT or Apache-2.0 without resolving the noncommercial upstream derivatives and restricted model code. Replacement or upstream permission may be required.

Before applying the final license, the authors should confirm ownership of new contributions, agreement of all relevant rights holders, scope of the chosen license, and any required permissions for GraspNet/FGC derivatives and scene assets. This candidate deliberately retains a license-review document instead of pretending that permission has already been granted.
