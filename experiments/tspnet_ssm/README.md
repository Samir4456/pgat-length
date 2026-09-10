# TSPNet + SSM Adapter (cross-backbone experiment)

Files here are the patched TSPNet code needed to insert the bidirectional
selective SSM adapter into TSPNet's fairseq encoder. They live in
`experiments/tspnet_ssm/` inside the pgat-length repo so they travel via
our existing GitHub -> cluster pipeline; on the cluster they get copied
into a fresh clone of TSPNet by `setup_cluster.sh`.

## Contents

| File | Role |
|---|---|
| `ssm_adapter.py` | The bidirectional selective SSM module (~180 lines, pure PyTorch). Same math as `pgat-length/src/pgat_length/models/ssm_adapter.py`, adjusted for `d_model=1024` (TSPNet's encoder dim). |
| `transformer_from_sign.py` | Fully patched TSPNet encoder. Adds `--use-ssm-adapter` CLI flag (opt-in; default off keeps published baseline) plus 6 hyperparameter flags; instantiates the adapter in `TransformerEncoderSign.__init__`; applies it in `forward` after the multi-scale concat, before the fairseq `(T, B, C)` transpose. |
| `run_phoenix_pos_embed_sp_test_3lvl_ssm.sh` | Runner script. Diff vs the baseline runner: appends `--use-ssm-adapter --ssm-d-state 16 --ssm-d-conv 4 --ssm-expand 2 --ssm-num-layers 1 --ssm-dropout 0.1 --ssm-residual-scale 1.0`. Nothing else changes. |
| `SSM_EXPERIMENT.md` | Full experimental protocol: env setup, data download, baseline reproduction, SSM training, evaluation, rescoring against the pgat-length scorer. |
| `setup_cluster.sh` | One-command cluster installer. Clones TSPNet upstream to `$HOME/tspnet-ssm`, copies the patched files in, prints the remaining manual steps. |

## Quick start (on the cluster)

```bash
cd $HOME/pgat-length && git pull
bash experiments/tspnet_ssm/setup_cluster.sh
```

Then follow the printed instructions (conda env, gdown data pull, two
training runs, two evaluations).

## Design choice: patched file, not a patch

`transformer_from_sign.py` here is the fully-patched version, not a git
patch. Rationale: shipping the whole file removes any risk of patch
context mismatches against different TSPNet checkouts. The overwrite
happens through `setup_cluster.sh` after a fresh upstream clone, so the
substitution is transparent.

If you diff `experiments/tspnet_ssm/transformer_from_sign.py` against
the upstream file, the changes are:

1. `from fairseq.models.ssm_adapter import BidirectionalSSMAdapter, SSMAdapterConfig`
2. `add_args`: `--use-ssm-adapter` and 6 hyperparameter flags.
3. `base_architecture`: defaults for the same 7 args.
4. `TransformerEncoderSign.__init__`: instantiates `self.ssm_adapter` when the flag is on.
5. `TransformerEncoderSign.forward`: applies the adapter to `(B, T_concat, C)` before the fairseq transpose.

Nothing else in the file is modified; when the flag is off the model is
byte-equivalent to the published TSPNet baseline.
