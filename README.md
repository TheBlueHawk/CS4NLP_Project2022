# MC-TACO 🌮
Dataset and code for “Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding EMNLP 2019. ([link](https://arxiv.org/abs/1909.03065))

## Dataset
We provide the dev/test split as specified in the paper, along with a detailed readme.txt file under `dataset/`

## Leaderboard 
See the details and instructions at: 
http://leaderboard.allenai.org/mctaco

## Experiments (WIP)
At this point, we provide the outputs of the ESIM/BERT baselines. 

To run BERT baseline: 

First install required packages with: 
```bash 
pip install -r experiments/bert/requirements.txt
```

### BERT

Your the following command to reproduce BERT predictions under `./bert_output`: 
```bash
sh experiments/bert/run_bert_baseline.sh
```
Evaluate the predictions with which you can further evaluate with the following command: 

```bash 
python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file bert_output/eval_outputs.txt
```

### RoBERTa
Your the following command to reproduce RoBERTa predictions under `./roberta_output`: 
```bash
sh experiments/roberta/run_roberta_baseline.sh
```

Evaluate the predictions with which you can further evaluate with the following command: 

```bash 
python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file roberta_output/eval_outputs_large.txt
```
Modify `run_roberta_baseline.sh` to switch between `roberta-base` and `roberta-large`.
### ESIM baseline: 
Releasing soon after some polish

## Citation
See the following paper:

```
@inproceedings{ZKNR19,
    author = {Ben Zhou, Daniel Khashabi, Qiang Ning and Dan Roth},
    title = {“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding },
    booktitle = {EMNLP},
    year = {2019},
}
```
