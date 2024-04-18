import argparse
import json
import ast
from collections import defaultdict
from typing import Optional

AUX_VERBS = set(["has", "hasn't", "have", "haven't"])
VERBS = set(["entertained", "amused", "applauded", "confused", "admired", "accepted", "remembered", "comforted", "annoyed",
             "giggled", "smiled", "slept", "swum", "waited", "moved", "changed", "read", "eaten"])

def parse_gold(gold: str) -> dict:
    components = {}
    words = gold.split()
    components["main_subject"] = " ".join(words[:2])
    components["distractors"] = []
    if words[2] not in AUX_VERBS:   # distractor phrase
        if words[2] in ("that", "who", "which"):    # complementizer
            distractor_seen = False
        else:
            distractor_seen = True
        distractor_phrase = []      # list of words
        for word in words[2:]:
            if word in AUX_VERBS:
                if not distractor_seen:
                    distractor_seen = True
                else:
                    main_aux = word
                    break
            distractor_phrase.append(word)
        components["distractors"].append(" ".join(distractor_phrase))
    else:
        main_aux = words[2]
    main_aux_idx = words.index(main_aux)
    main_vp = f"{main_aux} {words[main_aux_idx+1]}"
    components["main_aux"] = main_aux
    components["main_vp"] = main_vp.strip(".")
    if len(words) > main_aux_idx+2:     # there is an object
        components["main_object"] = " ".join(words[main_aux_idx+2:main_aux_idx+4]).strip(".")
        if len(words) > main_aux_idx+4:
            distractor_phrase = " ".join(words[main_aux_idx+4:]).strip(".")
            components["distractors"].append(distractor_phrase)
    else:
        components["main_object"] = ""
    return components


def parse_answer(answer: str) -> dict:
    components = {}
    words = answer.split()
    components["main_subject"] = " ".join(words[1:3]).strip("?")
    components["main_aux"] = words[0]
    components["distractors"] = []
    distractor_phrase = None
    main_v = None
    main_v_idx = None
    is_ungrammatical = False
    if len(words) < 4:
        return None, True
    if words[3] not in VERBS:   # distractor phrase
        if words[3] in ("that", "who", "which"):    # complementizer
            verb_seen = False
        else:
            verb_seen = True
        distractor_phrase = []      # list of words
        for idx in range(3, len(words)):
            word = words[idx]
            if word.strip("?") in VERBS:
                if not verb_seen:
                    verb_seen = True
                else:
                    main_v = word
                    main_v_idx = idx
                    break
            distractor_phrase.append(word)
        components["distractors"].append(" ".join(distractor_phrase))
    else:
        main_v = words[3]
        main_v_idx = 3

    if main_v is None:
        is_ungrammatical = True
        return components, is_ungrammatical
    main_vp = f"{components['main_aux']} {main_v}"
    components["main_vp"] = main_vp.strip("?")
    if len(words) > main_v_idx+1:     # there is an object
        components["main_object"] = " ".join(words[main_v_idx+1:main_v_idx+3]).strip("?")
        if len(words) > main_v_idx+3:
            distractor_phrase = " ".join(words[main_v_idx+3:]).strip("?")
            components["distractors"].append(distractor_phrase)
    else:
        components["main_object"] = ""
    return components, is_ungrammatical


def parse_pred(pred: str) -> dict:
    if "The answer is" in pred:
        pred_tokens = pred.split("The answer is")[0].split()
    else:
        pred_tokens = pred.split()
    components = {}
    curr_component = None
    curr_value = None
    is_list = False
    for idx, word in enumerate(pred_tokens):
        print(word)
        if word == "=":
            next_val = pred_tokens[idx+1]
            if next_val.startswith("\"") or next_val.startswith("["):
                curr_component = pred_tokens[idx-1]
            continue
        if curr_component is not None:
            if word.startswith("[") and word.endswith("]"):
                try:
                    components[curr_component] = ast.literal_eval(word)
                except:
                    pass
                curr_component = None
                continue
            if word.startswith("["):
                is_list = True
                curr_value = word
                continue
            if is_list:
                curr_value += f" {word}"
                if word.endswith("]"):
                    is_list = False
                    try:
                        components[curr_component] = ast.literal_eval(curr_value)
                    except:
                        pass
                    curr_component = None
                    curr_value = None
            else:
                if word.startswith("\"") and word.endswith("\""):
                    components[curr_component] = word.strip("\"")
                    curr_component = None
                    continue
                if word.startswith("\""):
                    curr_value = word.strip("\"")
                    continue
                if word.endswith("\""):
                    word = word.strip("\"")
                    curr_value += f" {word}"
                    components[curr_component] = curr_value
                    curr_component = None
                    curr_value = None
                    continue
                else:
                    curr_value += f" {word}"
    # get final answer
    if "The answer:" in pred:
        answer = pred.split("The answer:")[1].split("Q: ")[0].strip()
    elif "The answer is" in pred:
        answer = pred.split("The answer is")[1].split("Q:")[0].strip()
    else:
        answer = pred.strip()
    return components, answer


def detailed_eval(test_examples, preds) -> dict:
    total = 0
    seq_acc = 0
    main_aux_acc = 0
    component_accs = defaultdict(int)
    overall_acc = 0
    for test, pred in zip(test_examples, preds):
        total += 1
        test_data = json.loads(test)
        src, gold = test_data["src"], test_data["tgt"]
        # parse individual components from sentence
        src = src.split("question. ")[1].split("\n")[0].strip()
        gold_components = parse_gold(src)
        pred_components, pred_answer = parse_pred(pred)
        # evaluate individual components
        this_acc = 0
        max_score = 0
        for component in gold_components.keys():
            max_score += 1
            # before = component_accs[component]
            if component not in pred_components:
                continue
            if isinstance(gold_components[component], list):    # compute partial accuracy
                for item in gold_components[component]:
                    for item2 in pred_components[component]:
                        if item.lower() == item2.lower():
                            component_accs[component] += 1 / len(gold_components[component])
                            this_acc += 1 / len(gold_components[component])
            elif gold_components[component].lower() == pred_components[component].lower():
                component_accs[component] += 1
                this_acc += 1
            else:
                print(pred_components)
                print(pred_answer)
        # evaluate overall accuracy
        if pred_answer.lower() == gold.lower():
            seq_acc += 1
        if pred_answer.lower().split()[0] == gold.lower().split()[0]:
            main_aux_acc += 1
        if round(this_acc, 4) == max_score:
            overall_acc += 1

    # normalize counts to get accuracies
    for component in gold_components.keys():
        component_accs[component] /= total
    seq_acc /= total
    main_aux_acc /= total
    overall_acc /= total

    return (component_accs, overall_acc, seq_acc, main_aux_acc)
        

def faithfulness_eval(preds):
    total = 0
    total_grammatical = 0
    overall_faithfulness = 0
    ungrammatical = 0
    component_accs = defaultdict(int)
    for idx, pred in enumerate(preds):
        total += 1
        reasoning_components, pred_answer = parse_pred(pred)
        answer_components, is_ungrammatical = parse_answer(pred_answer)
        if not is_ungrammatical:
            total_grammatical += 1
        else:
            ungrammatical += 1
            continue
        max_score = 0
        faithful_components = 0
        for component in answer_components.keys():
            max_score += 1
            before = component_accs[component]
            if component not in reasoning_components:
                continue
            if isinstance(answer_components[component], list):    # compute partial accuracy
                for item in answer_components[component]:
                    for item2 in reasoning_components[component]:
                        if item.lower() == item2.lower():
                            component_accs[component] += 1 / len(reasoning_components[component])
                after = component_accs[component] - before
                faithful_components += after
            elif reasoning_components[component].lower() == answer_components[component].lower():
                component_accs[component] += 1
                faithful_components += 1
        # evaluate overall accuracy
        if faithful_components == max_score:
            overall_faithfulness += 1
         
    # normalize counts to get accuracies
    for component in component_accs.keys():
        component_accs[component] /= total_grammatical
    overall_faithfulness /= total_grammatical
    ungrammatical /= total

    return (component_accs, overall_faithfulness, ungrammatical)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, help="Path to test file containing line-separated JSON-formatted dictionaries.")
    parser.add_argument("preds_path", type=str, help="Path to predictions file.")
    args = parser.parse_args()

    with open(args.test_file, 'r') as test_examples, open(args.preds_path, 'r') as preds:
        acc_dict, overall_reasoningacc, seq_acc, main_aux_acc = detailed_eval(test_examples, preds)
        # reset index for reading file again
        preds.seek(0)
        faithful_dict, faithful, ungrammatical = faithfulness_eval(preds)
    
    print(f"Reasoning accuracy: {overall_reasoningacc:.2f}")
    print(f"Accuracy: {seq_acc} (exact match) / {main_aux_acc} (main aux)")
    for component in acc_dict:
        print(f"\t{component}: {acc_dict[component]}")
    
    print()
    print(f"Prediction/reasoning faithfulness: {faithful}")
    for component in faithful_dict:
        print(f"\t{component}: {faithful_dict[component]}")

    print()
    print(f"Ungrammatical: {ungrammatical}")
