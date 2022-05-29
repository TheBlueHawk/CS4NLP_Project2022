#!/usr/bin/env bash
set -e

# dry_run="-d"  # Uncomment this line to do partial run (helpful for checking that all questions are valid).

export SLURM_DISABLE_STATUS=1

# echo="echo"
ssrun="$echo srun -p jsteinhardt --gres=gpu:A100:1"   # (40 GB or 80 GB depending on machine).
# ssrun='srun -p jsteinhardt --gres=gpu:A100:1 -w balrog'  # (40 GB  balrog)
ssrun_big="$echo srun -p jsteinhardt --gres=gpu:A100:1 -w saruman" #  (80 GB  smaug)

# ssrun='srun -p jsteinhardt --gres=gpu:A5000:1'  # rainbowquartz (8)  (24 GB) (8x)
ssrun_small="$echo srun -p jsteinhardt --gres=gpu:A4000:1"  # smokyquartz, sunstone  (16 GB) (16x) (actually only 9x..)
# ssrun='srun -p jsteinhardt --gres=gpu:RTX8000:1'  # smaug  (48 GB) (2x)
log_base="runs/subset-tuning-final-fixed-binary/$(python scripts/make_uniq_run_directory.py)"

n_runs=5
n_shards=5

echo "Logging in log_base=$log_base"
# script="python src/maud/tune.py"

# TODO? Set a certain number of epochs based on helpful? OR maybe not... just run it!
# TODO: Find good hyperparameters first.
# for model in roberta-{large,base} microsoft/deberta-v3-base bert-base-cased; do
#
# Preferred learning rates:
#  $thing -m roberta-large -l 4e-06 --shard $shard_num/2 &
#  $thing -m roberta-base -l 1e-05 --shard $shard_num/2 &
#  $thing -m bert-base-cased -l 4e-06 --shard $shard_num/2 &

script="python src/maud/tune.py -A --num_updates 100:10 --skip_degenerate_jobs"

# # TODO(shwang): Try runs with synth capped later.
# # for flags in o oa oas os osc oasc; do
# for flags in o oa oas os; do
#   for shard_num_plus_one in $(seq $n_shards); do
#     shard_num=$((shard_num_plus_one - 1))
#
#     script2="$script --shard $shard_num/$n_shards -r $n_runs -$flags"
#
#     for lr in 1e-5 2e-5 3e-5; do
#       $ssrun $script2 -m roberta-large -l $lr --log_root $log_base/roberta-large-lr$lr/$flags &
#     done
#
#     for lr in 1e-05 2e-05 3e-05; do
#       $ssrun $script2 -m roberta-base -l $lr --log_root $log_base/roberta-base-lr$lr/$flags &
#     done
#
#     for lr in 2e-05 3e-05 5e-5; do
#       $ssrun_small $script2 -m bert-base-cased -l $lr --log_root $log_base/bert-base-cased-lr$lr/$flags &
#     done
#   done
# done
# wait

for flags in o oa oas os; do
  for shard_num_plus_one in $(seq $n_shards); do
    shard_num=$((shard_num_plus_one - 1))

    script2="$script --shard $shard_num/$n_shards -r $n_runs -$flags"

    # for lr in 5e-6 8e-6 9e-6 1e-5; do  # CHECK
    for lr in 5e-6 9e-6; do
      $ssrun_big $script2 -m microsoft/deberta-large -l $lr --log_root $log_base/deberta-large-lr$lr/$flags &
      $ssrun_big $script2 -m microsoft/deberta-v3-large -l $lr --log_root $log_base/deberta-v3-large-lr$lr/$flags &
    done

    # for lr in 1.5e-5 2e-5 3e-5 4e-5; do  # CHECK
    for lr in 1.5e-5 3e-5; do
      $ssrun $script2 -m microsoft/deberta-base -l $lr --log_root $log_base/deberta-base-lr$lr/$flags &
      $ssrun $script2 -m microsoft/deberta-v3-base -l $lr --log_root $log_base/deberta-v3-base-lr$lr/$flags &
    done
  done
done
wait
