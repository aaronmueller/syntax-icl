import os
import json
import openai
import argparse
import time

from tqdm import tqdm
from collections import defaultdict
from generation_utils import (
    get_openai_response, get_openai_response_chatmodels,
    load_llama, get_llama_response
)
from evaluation import (
    eval_transformation, eval_tense, eval_hans,
    evaluate_from_file, evaluate_example
)

def batchify(lines, batch_size):
    """Given list `lines`, return list of batches."""
    batched = []
    curr_batch = []
    for idx, example in enumerate(lines):
        curr_batch.append(json.loads(example))
        if (idx+1) % batch_size != 0:
            continue
        batched.append(curr_batch)
        curr_batch = []

    if len(curr_batch) > 0:
        batched.append(curr_batch)
    return batched


def backoff_response(prompts, sleep_len, max_len, model, use_chat_model):
    """
    Call OpenAI generation function with `sleep_len` seconds between calls.
    Increases `sleep_len` if rate errors are encountered.
    """
    preds = None
    while preds is None:
        try:
            time.sleep(sleep_len)
            if use_chat_model:
                preds = get_openai_response_chatmodels(prompts[0], max_len, model)
            else:
                preds = get_openai_response(prompts, max_len, model)
        except openai.error.RateLimitError:     # Increase sleep length if RateLimitErrors arise
            sleep_len += 5
            print(f"Rate limit error. Increasing sleep length to {sleep_len}s.")
        except openai.error.APIError:           # Random API errors
            print("API Error. Retrying...")
        # except openai.error.InvalidRequestError:    # Rarely, the API refuses to answer. Return boilerplate if so
        #     err_msg = "Content filtered! Returning this message as prediction."
        #     print(err_msg)
        #     return [{"text": err_msg}] * len(prompts), sleep_len
    return preds, sleep_len


def generate_and_evaluate(preds_filepath, test_filepath, model_name, llama,
                          task, batch_size, max_len, init_sleep_len,
                          print_errors=False, use_chat_model=False):
    if llama:
        llama_tokenizer, llama_model = load_llama(model_name)
    
    correct = 0
    total = 0
    correct_grouped = defaultdict(lambda: defaultdict(int))    # for HANS
    total_grouped = defaultdict(lambda: defaultdict(int))
    sleep_len = init_sleep_len
    with open(preds_filepath, "w") as preds_file, open(test_filepath, "r") as test_file:
        lines = test_file.readlines()
        batched = batchify(lines, batch_size)
        for batch in tqdm(batched, desc="Batches", total=len(batched)):
            inputs = []
            for item in batch:
                src = item["src"]
                formatted_src = prompt + src
                inputs.append(formatted_src)
            # Generate predictions
            if llama:
                preds = get_llama_response(llama_tokenizer, llama_model, inputs, max_len)
                preds = [" ".join(preds[idx].split()[len(_input.split()):]) for idx, _input in enumerate(inputs)]
            else:
                preds, sleep_len = backoff_response(inputs, sleep_len, max_len,
                                                    model_name, use_chat_model)
            # iterate through each prediction in batch
            for item, pred_dict in zip(batch, preds):
                tgt = item["tgt"]
                if use_chat_model:
                    pred = pred_dict["message"]["content"]
                elif llama:
                    pred = pred_dict
                else:
                    pred = pred_dict["text"]
                # Write prediction to file
                preds_file.write(pred.replace("\n", " ") +"\n")
                total += 1

                diff_idx = None if "diff_idx" not in item else item["diff_idx"]
                
                # Evaluate whether prediction is correct
                correct_bool = evaluate_example(pred, tgt, task,
                                                diff_idx=diff_idx, print_errors=print_errors)
                correct += int(correct_bool)
                if task == "hans":
                    correct_grouped[item["heuristic"]][tgt] += int(correct_bool)
                    total_grouped[item["heuristic"]][tgt] += 1
                if print_errors and not correct_bool:
                    print(tgt, "\t|||\t", pred)

    return correct, correct_grouped, total, total_grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Evaluation task.")
    parser.add_argument("prompt", type=str, help="Path to exemplar string as .txt file.")
    parser.add_argument("--test_file", "-t", type=str, default="test.ood.json", help="Test file as .json.")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size for API calls.") 
    parser.add_argument("--max_len", "-l", type=int, default=256,
                        help="Maximum number of tokens the API should return for each request.")
    parser.add_argument("--init_sleep_len", "-s", type=float, default=1.0,
                        help="Amount of time between requests.")
    parser.add_argument("--eval_only", "-e", action="store_true",
                        help="No API calls or generation; just evaluate.")
    parser.add_argument("--oai_model", "-m", type=str, default="code-davinci-002", help="Name of model.")
    parser.add_argument("--use_gpt35", "-3", action="store_true", help="Use GPT-3.5-Turbo.")
    parser.add_argument("--use_gpt4", "-4", action="store_true", help="Use GPT-4.")
    parser.add_argument("--llama", type=str, default=None, help="Use Llama 2 model.")
    parser.add_argument("--print_errors", "-p", action="store_true", help="Print incorrect outputs.")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    use_chat_model = False
    if args.use_gpt35 or args.use_gpt4:
        assert args.batch_size == 1     # Can't use batching with chat models
        use_chat_model = True
    # Prevent unexpected behaviors - no model conflicts
    assert not (args.use_gpt35 and args.use_gpt4)

    # Get model name
    if args.llama:
        model_name = args.llama
    else:
        model_name = "gpt-3.5-turbo" if args.use_gpt35 else "gpt-4" if args.use_gpt4 else args.oai_model

    # Make path to predictions directory
    out_dir = f"data/{args.task}/preds/"
    if args.use_gpt35:
        out_dir += "gpt35/"
    elif args.use_gpt4:
        out_dir += "gpt4/"
    else:
        out_dir += f"{model_name}/"

    # Make full path to predictions and test file
    preds_path = f"{out_dir}/{os.path.splitext(args.prompt)[0]}.{os.path.splitext(args.test_file)[0]}.preds"
    out_dir = os.path.dirname(preds_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    test_path = f"data/{args.task}/{args.test_file}"

    # Don't run generation if this arg is set
    if args.eval_only:
        correct, correct_grouped, total, total_grouped = evaluate_from_file(preds_path, test_path, args.task,
                                                                            args.print_errors)

    # Else, generate predictions, too
    else:
        with open(f"data/{args.task}/{args.prompt}", "r") as prompt_file:
            prompt = prompt_file.read()
        while not prompt.endswith("\n\n"):
            prompt += "\n"
        
        correct, correct_grouped, total, total_grouped = generate_and_evaluate(
            preds_path, test_path, model_name, args.llama, args.task,
            args.batch_size, args.max_len, args.init_sleep_len,
            print_errors=args.print_errors, use_chat_model=use_chat_model
        )
        
    # Display overall accuracies
    print(f"Accuracy: {correct / total}")

    # If HANS, break down accuracies by heuristic type and label
    if args.task == "hans":
        for heuristic in correct_grouped:
            for entailment in correct_grouped[heuristic]:
                acc = correct_grouped[heuristic][entailment] / total_grouped[heuristic][entailment]
                print(f"Accuracy ({heuristic}) -- {entailment}: {acc}")
