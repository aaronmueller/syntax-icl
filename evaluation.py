import json
from collections import defaultdict

def eval_transformation(pred, gold, transformation_type="question"):
    """Evaluate a single question formation output."""
    gold = gold.lower().strip()
    pred = pred.lower().strip()
    idx = 0 if transformation_type == "question" else 1
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    if len(pred) < idx+1:
        return False
    return pred.split()[idx] == gold.split()[idx]


def eval_tense(pred, gold, diff_idx):
    """Evaluate a single tense reinflection output."""
    gold = gold.lower().strip()
    pred = pred.lower().strip()
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    correct_list = [p == g for p, g in zip(pred.split(), gold.split())]
    # correct only if all verbs are correct
    return all(correct_list)


def eval_hans(pred, gold):
    """Evaluate a single HANS output."""
    gold = gold.lower()
    pred = pred.lower()
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    pred = pred.split()[0].strip(".").strip(",")
    return pred == gold


def evaluate_example(pred, tgt, task, diff_idx=None, print_errors=False):
    if task == "hans":
        correct_bool = eval_hans(pred, tgt)
    elif task == "question_formation":
        correct_bool = eval_transformation(pred, tgt)
    elif task == "tense_reinflection":
        correct_bool = eval_tense(pred, tgt, diff_idx)
    return correct_bool


def evaluate_from_file(preds_filepath, test_filepath, task, print_errors=False):
    """
    Evaluate all outputs predictions for a given testset and file containing
    model predictions on that testset.
    """
    correct_grouped = defaultdict(lambda: defaultdict(int))
    total_grouped = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    # iterate through pairs of aligned examples and outputs
    with open(preds_filepath, "r") as preds_file, open(test_filepath, "r") as test_file:
        for gold, pred in zip(test_file, preds_file):
            total += 1
            pred = pred.strip()
            data = json.loads(gold)
            tgt = data["tgt"]
            diff_idx = None if "diff_idx" not in data else data["diff_idx"]
            correct_bool = evaluate_example(pred, tgt, task,
                                            diff_idx=diff_idx, print_errors=print_errors)
            correct += int(correct_bool)
            if task == "hans":  # group results by label and heuristic type
                correct_grouped[data["heuristic"]][str(data["tgt"])] += int(correct_bool)
                total_grouped[data["heuristic"]][str(data["tgt"])] += 1
            if print_errors and not correct_bool:
                print(tgt, "\t|||\t", pred)
    
    return correct, correct_grouped, total, total_grouped
