import argparse
import json

"""
Example Usage: 
python evaluator/auto_evaluator.py eval --model_name bert-base-uncased
"""


class McTacoEvaluator:

    def __init__(self, model_name):
        self.test_file = "dataset/test_9442.tsv"
        self.output_file = "outputs/" + model_name + "/eval_outputs.txt"
        self.output = "outputs/" + model_name + "/test_accuracy.txt"

    def print_result(self):
        ref_lines = [x.strip() for x in open(self.test_file).readlines()]
        prediction_lines = [x.strip() for x in open(self.output_file).readlines()]

        result_map = {}
        prediction_count_map = {}
        prediction_map = {}
        gold_count_map = {}
        for i, line in enumerate(ref_lines):
            key = " ".join(line.split("\t")[0:2])
            if key not in result_map:
                result_map[key] = []
                prediction_count_map[key] = 0.0
                gold_count_map[key] = 0.0
                prediction_map[key] = []
            prediction = prediction_lines[i]
            prediction_map[key].append(prediction)
            label = line.split("\t")[3]
            if prediction == "yes":
                prediction_count_map[key] += 1.0
            if label == "yes":
                gold_count_map[key] += 1.0
            result_map[key].append(prediction == label)

        total = 0.0
        correct = 0.0
        f1 = 0.0
        for question in result_map:
            val = True
            total += 1.0
            cur_correct = 0.0
            for i, v in enumerate(result_map[question]):
                val = val and v
                if v and prediction_map[question][i] == "yes":
                    cur_correct += 1.0
            if val:
                correct += 1.0
            p = 1.0
            if prediction_count_map[question] > 0.0:
                p = cur_correct / prediction_count_map[question]
            r = 1.0
            if gold_count_map[question] > 0.0:
                r = cur_correct / gold_count_map[question]
            if p + r > 0.0:
                f1 += 2 * p * r / (p + r)

        print("Strict Acc.: " + str(correct / total))
        print("Avg F1: " + str(f1 / total))

        if self.output:
            print("Writing results to file: %s" % self.output)
            with open(self.output, "wt", encoding="UTF-8") as output:
                output.write(json.dumps({
                    "em": correct / total,
                    "f1": f1 / total
                }))

    def print_errors(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="[eval]")
    parser.add_argument("--model_name",
                        required=True,
                        help="model name as listed on HuggingFace.")
    

    args = parser.parse_args()
    print(args.model_name)
    if args.command == "eval":
        evaluator = McTacoEvaluator(args.model_name)
        evaluator.print_result()
    else:
        print("Command not found, see --help.")


if __name__ == "__main__":
    main()