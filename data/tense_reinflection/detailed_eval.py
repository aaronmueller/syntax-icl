import argparse
import json
import ast
from collections import defaultdict
from typing import Optional

SINGULAR_NOUNS = set(["zebra", "yak", "newt", "tyrannosaurus", "unicorn", "peacock", "salamander", "orangutan", "walrus", "vulture", "xylophone", "quail", "raven"])
PAST_VERBS = set(["entertained", "amused", "applauded", "confused", "admired", "accepted", "remembered", "comforted", "annoyed",
             "giggled", "smiled", "slept", "swum", "swam", "waited", "moved", "changed", "read", "eaten", "ate"])
COMPLEMENTIZERS = set(["who", "that"])
PREPOSITIONS = set(["near", "behind", "around", "below", "above", "by", "upon", "with"])

def make_present_verbs(past_verbs):
    singular_verbs = set()
    plural_verbs = set()
    remove_ed_list = set(["accepted", "remembered", "comforted", "annoyed", "waited", "entertained",
                          "applauded"])
    irregulars = {"slept": ["sleeps", "sleep"],
                  "swam": ["swims", "swim"],
                  "read": ["reads", "read"],
                  "ate": ["eats", "eat"]}
    for verb in past_verbs:
        if verb in irregulars:
            singular = irregulars[verb][0]
            plural = irregulars[verb][1]
        elif verb in remove_ed_list:
            singular = verb.replace("ed", "s")
            plural = verb.replace("ed", "")
        else:
            singular = verb.replace("ed", "es")
            plural = verb.replace("ed", "e")
        singular_verbs.add(singular)
        plural_verbs.add(plural)
    return (singular_verbs, plural_verbs)


def make_plural_nouns(singular_nouns):
    plural_nouns = set()
    for noun in singular_nouns:
        if noun.endswith("s"):
            plural = noun + "es"
        else:
            plural = noun + "s"
        plural_nouns.add(plural)
    return plural_nouns


def parse_np(noun, noun_idx, words, has_main_verb=True):
    """Returns dictionary with nouns as keys and verbs as values.
       Also returns the index of the main verb."""
    # noun is the last word in the sentence
    if len(words)-1 == noun_idx:
        subjects_verbs = {noun: []}
        main_verb_idx = None
    # (NP) (VP)
    elif words[noun_idx+1] in VERBS:
        main_verb_idx = noun_idx+1
        subjects_verbs = {noun: [words[main_verb_idx]]}
    # (NP (CP (NP VP))) (VP) or (NP (CP (VP NP))) (VP) or (NP (CP (VP))) (VP)
    elif words[noun_idx+1] in COMPLEMENTIZERS:
        # (NP (CP (VP NP))) (VP)
        if words[noun_idx+2] in VERBS and (len(words) > noun_idx+3 and words[noun_idx+4] in NOUNS):
            relcl_verb = words[noun_idx+2]
            relcl_obj = words[noun_idx+4]
            main_verb_idx = noun_idx+5 if has_main_verb else None
            if has_main_verb:
                subjects_verbs = {noun: [relcl_verb, words[main_verb_idx]], relcl_obj: []}
            else:
                subjects_verbs = {noun: [relcl_verb], relcl_obj: []}
        # (NP (CP (VP))) (VP)
        elif words[noun_idx+2] in VERBS:
            relcl_verb = words[noun_idx+2]
            main_verb_idx = noun_idx+3 if has_main_verb else None
            if has_main_verb:
                subjects_verbs = {noun: [relcl_verb, words[main_verb_idx]]}
            else:
                subjects_verbs = {noun: [relcl_verb]}
        # (NP (CP (NP VP))) (VP)
        elif words[noun_idx+3] in NOUNS:
            relcl_noun = words[noun_idx+3]
            relcl_verb = words[noun_idx+4]
            main_verb_idx = noun_idx+5 if has_main_verb else None
            if has_main_verb:
                subjects_verbs = {noun: [words[main_verb_idx]], relcl_noun: [relcl_verb]}
            else:
                subjects_verbs = {noun: [], relcl_noun: [relcl_verb]}
    # (NP (PP (P NP))) (VP)
    elif words[noun_idx+1] in PREPOSITIONS:
        pp_noun = words[noun_idx+3]
        main_verb_idx = noun_idx+4 if has_main_verb else None
        if has_main_verb:
            subjects_verbs = {noun: [words[main_verb_idx]], pp_noun: []}
        else:
            subjects_verbs = {noun: [], pp_noun: []}
    else:
        raise Exception(f"Cannot parse string: {' '.join(words)}")

    return (subjects_verbs, main_verb_idx)


def parse_gold(gold: str) -> dict:
    components = {}
    words = gold.split()
    words[-1] = words[-1].strip().replace(".", "")
    components["nouns"] = []
    components["subjects_verbs"] = {}
    components["sentence"] = " ".join(words).lower()

    main_subj_idx = 1
    main_subj = words[main_subj_idx]
    main_subj_parsed, main_verb_idx = parse_np(main_subj, main_subj_idx, words)
    components["subjects_verbs"].update(main_subj_parsed)

    if len(words) > main_verb_idx+1:
        obj_index = main_verb_idx + 2
        obj = words[obj_index]
        main_obj_parsed, _ = parse_np(obj, obj_index, words, has_main_verb=False)
        for subject in main_obj_parsed:
            if subject in components["subjects_verbs"]:
                components["subjects_verbs"][subject].extend(main_obj_parsed[subject])
            else:
                components["subjects_verbs"][subject] = main_obj_parsed[subject]

    # postprocess dictionary
    components["nouns"] = [item for item in components["subjects_verbs"].keys()]

    return components


def parse_answer(answer: str) -> dict:
    components = {}
    words = answer.split()
    words[-1] = words[-1].strip().replace(".", "")
    is_ungrammatical = False

    components["nouns"] = []
    components["subjects_verbs"] = {}
    components["sentence"] = " ".join(words).lower()

    main_subj_idx = 1
    main_subj = words[main_subj_idx]
    main_subj_parsed, main_verb_idx = parse_np(main_subj, main_subj_idx, words)
    components["subjects_verbs"].update(main_subj_parsed)

    if len(words) > main_verb_idx+1:
        obj_index = main_verb_idx + 2
        obj = words[obj_index]
        try:
            main_obj_parsed, _ = parse_np(obj, obj_index, words, has_main_verb=False)
            for subject in main_obj_parsed:
                if subject in components["subjects_verbs"]:
                    components["subjects_verbs"][subject].extend(main_obj_parsed[subject])
                else:
                    components["subjects_verbs"][subject] = main_obj_parsed[subject]
        except:
            is_ungrammatical = True

    # postprocess dictionary
    components["nouns"] = [item for item in components["subjects_verbs"].keys()]
    
    return components, is_ungrammatical


def parse_pred(pred: str) -> dict:
    pred_tokens = pred.split()
    components = {}
    curr_component = None
    curr_value = None
    is_list = False
    is_dict = False
    for idx, word in enumerate(pred_tokens):
        if word == "=":
            if len(pred_tokens) <= idx+1:
                break
            next_val = pred_tokens[idx+1]
            if next_val.startswith("\"") or next_val.startswith("[") or next_val.startswith("{"):
                curr_component = pred_tokens[idx-1]
            continue
        if curr_component is not None:
            if word.startswith("[") and word.endswith("]"):
                components[curr_component] = ast.literal_eval(word)
                curr_component = None
                continue
            if word.startswith("[") and not is_dict:
                is_list = True
                curr_value = word
                continue
            if is_list:
                curr_value += f" {word}"
                if word.endswith("]"):
                    is_list = False
                    components[curr_component] = ast.literal_eval(curr_value)
                    curr_component = None
                    curr_value = None
            if word.startswith("{") and not is_list:
                is_dict = True
                curr_value = word
                continue
            if is_dict:
                curr_value += f" {word}"
                if word.endswith("}"):
                    is_dict = False
                    components[curr_component] = ast.literal_eval(curr_value)
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
    # get nouns from subjects_verbs dict
    components["expected_verbs"] = []
    if "subjects_verbs" in components:
        for subject in components["subjects_verbs"]:
            subject = subject.lower().replace(".", "")
            components["subjects_verbs"][subject] = [verb.lower().replace(".", "") for verb in \
                                                    components["subjects_verbs"][subject]]
            verb_list = components["subjects_verbs"][subject]
            if len(subject.split()) > 1:
                subject = subject.split()[-1]
            for verb in verb_list:
                if subject in SINGULAR_NOUNS:
                    components["expected_verbs"].append(make_present_verbs([verb])[0].pop())
                elif subject in PLURAL_NOUNS:
                    components["expected_verbs"].append(make_present_verbs([verb])[1].pop())
                else:
                    raise ValueError(f"Unrecognized subject: {subject}")

        components["nouns"] = [item for item in components["subjects_verbs"].keys()]
    else:
        components["nouns"] = []
    if "sentence" in components:
        components["sentence"] = components["sentence"].lower().strip().replace(".", "")
    

    """
    # GPT-4 outputs non-standard things. Here are some ad-hoc fixes that may help.
    elif "The present tense of" in pred:
        answer = pred.split("\"")[-2]
        components = None
    elif "The sentence in present" in pred:
        answer = pred.split("\"")[-2]
        components = None
    elif "to present tense:" in pred:
        answer = pred.split(":")[-1]
        components = None
    elif "already in present" in pred or "already in the present" in pred:
        return None, None
    """
    # get final answer
    answer = pred
    if "The answer is" in pred:
        answer = pred.split("The answer is ")[1]
    else:
        print(f"Warning: didn't find 'The answer is' in prediction. Prediction: {pred}")
        # components = None
        print()
    if "Q:" in answer:
        answer = answer.split("Q:")[0]
    answer = answer.strip()
    return components, answer


def detailed_eval(test_examples, preds) -> dict:
    total = 0
    seq_acc = 0
    main_aux_acc = 0
    overall_reasoningacc = 0
    component_accs = defaultdict(int)
    for test, pred in zip(test_examples, preds):
        total += 1
        test_data = json.loads(test)
        src, gold = test_data["src"], test_data["tgt"]
        # parse individual components from sentence
        src = src.split("Q: ")[1].split("\n")[0].strip()
        gold_components = parse_gold(src)
        pred_components, pred_answer = parse_pred(pred)
        if pred_components is None:
            total -= 1
            continue
        # evaluate individual components
        max_acc = 0
        accurate_components = 0
        for component in gold_components.keys():
            if component not in pred_components:
                continue
            max_acc += 1
            before = component_accs[component]
            if isinstance(gold_components[component], list):    # compute partial accuracy
                for item in gold_components[component]:
                    for item2 in pred_components[component]:
                        if item.lower() == item2.lower():
                            component_accs[component] += 1 / len(gold_components[component])
                after = component_accs[component] - before
                if after > 0.999:
                    component_accs[component] += 1 - after
                else:
                    print(gold_components, "|||", pred_components)
                accurate_components += after
            elif isinstance(gold_components[component], dict):  # compute partial accuracy
                for key in gold_components[component]:
                    if key not in pred_components[component]:
                        continue
                    if pred_components[component][key] == gold_components[component][key]:
                        component_accs[component] += 1 / len(gold_components[component].keys())
                after = component_accs[component] - before
                if after > 0.999:
                    component_accs[component] += 1 - after
                accurate_components += after
            elif gold_components[component].lower() == pred_components[component].lower():
                component_accs[component] += 1
                accurate_components += 1
        # evaluate overall accuracy
        # sequence accuracy
        if pred_answer.lower() == gold.lower():
            seq_acc += 1
        # verb accuracy
        if pred_answer.lower().split()[0] == gold.lower().split()[0]:
            main_aux_acc += 1
        
        if round(accurate_components, 5) == max_acc:
            overall_reasoningacc += 1
        else:
            print(accurate_components, max_acc)
            print(gold_components, "|||", pred_components)

    # normalize counts to get accuracies
    for component in gold_components.keys():
        component_accs[component] /= total
    seq_acc /= total
    main_aux_acc /= total
    overall_reasoningacc /= total

    return (component_accs, overall_reasoningacc, main_aux_acc)


def make_present(subjects_verbs):
    new_subjects_verbs = {}
    for subject in subjects_verbs:
        singular_verbs, plural_verbs = make_present_verbs(subjects_verbs[subject])
        new_subjects_verbs[subject] = list(singular_verbs)
        new_subjects_verbs[subject].extend(list(plural_verbs))
    return new_subjects_verbs


def faithfulness_eval(test_examples, preds):
    total = 0
    total_grammatical = 0
    overall_faithfulness = 0
    ungrammatical = 0
    component_accs = defaultdict(int)
    no_reasoning = defaultdict(int)
    not_one = defaultdict(int)
    for test, pred in zip(test_examples, preds):
        total += 1
        reasoning_components, pred_answer = parse_pred(pred)
        if reasoning_components is None:
            total -= 1
            continue
        try:
            answer_components, is_ungrammatical = parse_answer(pred_answer)
        except:
            continue
        if not is_ungrammatical:
            total_grammatical += 1
        else:
            ungrammatical += 1
            continue
        max_score = 0
        faithful_components = 0
        for component in answer_components.keys():
            if component == "sentence":
                continue
            if component not in reasoning_components:
                no_reasoning[component] += 1
                continue
            max_score += 1
            before = component_accs[component]
            if isinstance(answer_components[component], list):    # compute partial accuracy
                for item in answer_components[component]:
                    for item2 in reasoning_components[component]:
                        if item.lower() == item2.lower():
                            component_accs[component] += 1 / len(answer_components[component])
                after = component_accs[component] - before
                if after > 0.999:
                    component_accs[component] += 1 - after
                else:
                    not_one[component] += 1
                faithful_components += after
            elif isinstance(answer_components[component], dict):  # compute partial accuracy
                reasoning_components[component] = make_present(reasoning_components[component])
                for key in answer_components[component]:
                    if key not in reasoning_components[component]:
                        continue
                    if len(reasoning_components[component][key]) == 0 and len(answer_components[component][key]) == 0:
                        component_accs[component] += 1 / len(reasoning_components[component])
                    else:
                        for value in answer_components[component][key]:
                            if value in reasoning_components[component][key]:
                                component_accs[component] += (1 / len(reasoning_components[component].keys())) / \
                                    (len(reasoning_components[component][key])/2)
                after = component_accs[component] - before
                if after > 0.999:
                    component_accs[component] += 1 - after
                else:
                    not_one[component] += 1
                faithful_components += after
            elif reasoning_components[component].lower() == answer_components[component].lower():
                component_accs[component] += 1
                faithful_components += 1

        # see whether all expected verbs are present
        max_score += 1
        test_data = json.loads(test)
        before = component_accs["expected_verbs"]
        for verb in reasoning_components["expected_verbs"]:
            if verb in answer_components["sentence"].strip().lower().replace(".", "").split():
                component_accs["expected_verbs"] += 1 / len(reasoning_components["expected_verbs"])
        after = component_accs["expected_verbs"] - before
        if after > 0.999:
            component_accs["expected_verbs"] += 1 - after
        else:
            not_one["expected_verbs"] += 1
        faithful_components += after

        # evaluate overall accuracy
        if round(faithful_components, 4) == max_score:
            overall_faithfulness += 1

    # normalize counts to get accuracies
    no_reasoning_total = max([no_reasoning[component] for component in no_reasoning])
    for component in component_accs.keys():
        component_accs[component] /= (total_grammatical - no_reasoning_total)
        if component in no_reasoning:
            no_reasoning[component] /= total_grammatical
    overall_faithfulness /= (total_grammatical - no_reasoning_total)
    ungrammatical /= total

    return (component_accs, overall_faithfulness, ungrammatical, no_reasoning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, help="Path to test file containing line-separated JSON-formatted dictionaries.")
    parser.add_argument("preds_path", type=str, help="Path to predictions file.")
    args = parser.parse_args()

    PLURAL_NOUNS = make_plural_nouns(SINGULAR_NOUNS)
    SINGULAR_VERBS, PLURAL_VERBS = make_present_verbs(PAST_VERBS)
    PRESENT_VERBS = SINGULAR_VERBS.union(PLURAL_VERBS)
    VERBS = PRESENT_VERBS.union(PAST_VERBS)
    NOUNS = SINGULAR_NOUNS.union(PLURAL_NOUNS)

    with open(args.test_file, 'r') as test_examples, open(args.preds_path, 'r') as preds:
        acc_dict, reasoning_acc, main_aux_acc = detailed_eval(test_examples, preds)
        # reset index for reading file again
        preds.seek(0)
        test_examples.seek(0)
        faithful_dict, faithful, ungrammatical, no_reasoning = faithfulness_eval(test_examples, preds)
    
    print(f"Reasoning Accuracy: {reasoning_acc}")
    print(f"Accuracy: {main_aux_acc} (main aux)")
    for component in acc_dict:
        print(f"\t{component}: {acc_dict[component]:.3f} (No reasoning: {no_reasoning[component]:.3f})")
    
    print()
    print(f"Prediction/reasoning faithfulness: {faithful}")
    for component in faithful_dict:
        print(f"\t{component}: {faithful_dict[component]:.3f} (No reasoning: {no_reasoning[component]:.3f})")

    print()
    print(f"Ungrammatical: {ungrammatical}")
