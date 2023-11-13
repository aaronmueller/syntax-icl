import os
import json
import openai
import argparse
import time

from tqdm import tqdm
from collections import defaultdict

def eval_transformation(pred, gold, transformation_type="question"):
    gold = gold.lower().strip()
    pred = pred.lower().strip()
    idx = 0 if transformation_type == "question" else 1
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    if len(pred) < idx+1:
        return False
    return pred.split()[idx] == gold.split()[idx]

def eval_tense(pred, gold, diff_idx):
    gold = gold.lower().strip()
    pred = pred.lower().strip()
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    correct_list = [p == g for p, g in zip(pred.split(), gold.split())]
    # correct only if all verbs are correct
    return all(correct_list)

def eval_hans(pred, gold):
    gold = gold.lower()
    pred = pred.lower()
    if "the answer is" in pred:
        pred = pred.split("the answer is")[1]
    pred = pred.split()[0].strip(".")
    return pred == gold

def batchify(lines, batch_size):
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

def get_openai_response(prompts, max_len, model):
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=max_len,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    # print(response["choices"])
    # return response["choices"]

def get_openai_response_chatmodels(prompt, max_len, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=max_len,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"]


def get_llama_response(tokenizer, model, prompts, max_len):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids,
                            top_p=0.9,
                            temperature=0.1,
                            max_new_tokens=max_len,
                            pad_token_id=tokenizer.eos_token_id)
    strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return strings


def backoff_response(prompts, sleep_len, max_len, model, use_chat_model):
    preds = None
    while preds is None:
        #try:
        time.sleep(sleep_len)
        if use_chat_model:
            preds = get_openai_response_chatmodel(prompts[0], max_len, model)
        else:
            preds = get_openai_response(prompts, max_len, model)
        except openai.error.RateLimitError:
            sleep_len += 5
            print(f"Rate limit error. Increasing sleep length to {sleep_len}s.")
        except openai.error.APIError:
            print("API Error. Retrying...")
        except openai.error.InvalidRequestError:
            err_msg = "Content filtered! Returning this message as prediction."
            print(err_msg)
            return [{"text": err_msg}] * len(prompts), sleep_len
    return preds, sleep_len


openai.api_key = os.getenv("OPENAI_API_KEY")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Evaluation task.")
    parser.add_argument("prompt", type=str, help="Path to exemplar string as .txt file.")
    parser.add_argument("--test_file", "-t", type=str, default="test.ood.json", help="Test file as .json.")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size for API calls.") 
    parser.add_argument("--max_len", "-l", type=int, default=256, help="Maximum number of tokens the API should return for each request.")
    parser.add_argument("--init_sleep_len", "-s", type=float, default=1.0, help="Amount of time between requests.")
    parser.add_argument("--eval_only", "-e", action="store_true", help="No API calls; just evaluation.")
    parser.add_argument("--model", "-m", type=str, default="code-davinci-002", help="Name of model.")
    parser.add_argument("--use_gpt35", "-g", action="store_true", help="Use GPT-3.5-Turbo.")
    parser.add_argument("--use_gpt4", "-4", action="store_true", help="Use GPT-4.")
    parser.add_argument("--llama", type=str, default=None, help="Use Llama 2 model.")
    parser.add_argument("--print_errors", "-p", action="store_true", help="Print incorrect outputs.")
    args = parser.parse_args()

    use_chat_model = False
    if args.use_gpt35 or args.use_gpt4:
        assert args.batch_size == 1     # Can't use batching with ChatGPT models
        use_chat_model = True
    # Prevent unexpected behaviors - no model conflicts
    assert not (args.use_gpt35 and args.use_gpt4)

    # Get model name
    model = "gpt-3.5-turbo-0301" if args.use_gpt35 else "gpt-4-0314" if args.use_gpt4 else args.model

    # Make path to predictions directory
    out_dir = f"data/{args.task}/preds/"
    if args.use_gpt35:
        out_dir += "gpt35/"
    elif args.use_gpt4:
        out_dir += "gpt4/"
    elif args.llama:    # load Llama model
        import torch
        out_dir += f"{args.llama}/"
        if not args.eval_only:
            if args.llama.startswith("meta-llama/Llama-2"):
                from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
                llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama)
                bnb_config = BitsAndBytesConfig(    # use 4-bit quantization to make it fit on a single GPU
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                llama_model = LlamaForCausalLM.from_pretrained(
                    args.llama,
                    trust_remote_code=True,
                    quantization_config=bnb_config
                )
            elif args.llama.startswith("codellama/"):
                from transformers import CodeLlamaTokenizer, AutoModelForCausalLM
                llama_tokenizer = CodeLlamaTokenizer.from_pretrained(args.llama)
                llama_model = AutoModelForCausalLM.from_pretrained(     # load in half-precision
                    args.llama,
                    torch_dtype=torch.bfloat16
                ).to("cuda")
    else:
        out_dir += f"{model}/"

    # Make full path to predictions
    out_name = f"{out_dir}/{os.path.splitext(args.prompt)[0]}.{os.path.splitext(args.test_file)[0]}.preds"
    out_dir = os.path.dirname(out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Don't run generation if this arg is set
    if args.eval_only:
        with open(out_name, "r") as preds_file, open(f"data/{args.task}/{args.test_file}", "r") as test_file:
            correct_grouped = defaultdict(lambda: defaultdict(int))
            total_grouped = defaultdict(lambda: defaultdict(int))
            total = 0
            correct = 0
            for gold, pred in zip(test_file, preds_file):
                total += 1
                pred = pred.strip()
                data = json.loads(gold)
                tgt = data["tgt"]
                if args.task == "hans":
                    correct_bool = eval_hans(pred, tgt)
                    correct_grouped[data["heuristic"]][str(data["tgt"])] += int(correct_bool)
                    total_grouped[data["heuristic"]][str(data["tgt"])] += 1
                    correct += int(correct_bool)
                    if not correct_bool:
                        print(tgt, "\t|||\t", pred)
                elif args.task == "transformations":
                    correct_bool = eval_transformation(pred, tgt)
                    correct += int(correct_bool)
                    if not correct_bool:
                        print(tgt, "\t|||\t", pred)
                elif args.task == "passivization":
                    correct += int(eval_transformation(pred, tgt, transformation_type="passivization"))
                elif args.task == "tense":
                    correct_bool = eval_tense(pred, tgt, data["diff_idx"])
                    correct += int(correct_bool)
                    if not correct_bool:
                        print(tgt, "\t|||\t", pred)
    # Else, generate predictions, too
    else:
        with open(f"data/{args.task}/{args.prompt}", "r") as prompt_file:
            prompt = prompt_file.read()
        while not prompt.endswith("\n\n"):
            prompt += "\n"
        
        sleep_len = args.init_sleep_len
        with open(f"data/{args.task}/{args.test_file}", "r") as test_file, open(out_name, "w") as preds_file:
            lines = test_file.readlines()
            batched = batchify(lines, args.batch_size)
            correct = 0
            correct_grouped = defaultdict(lambda: defaultdict(int))    # for HANS
            total_grouped = defaultdict(lambda: defaultdict(int))
            total = 0
            for batch in tqdm(batched):
                inputs = []
                for item in batch:
                    src = item["src"]
                    formatted_src = prompt + src
                    inputs.append(formatted_src)
                # Get predictions from API; increase sleep length if RateLimitErrors arise
                if args.llama:
                    preds = get_llama_response(llama_tokenizer, llama_model, inputs, args.max_len)
                    preds = [" ".join(preds[idx].split()[len(_input.split()):]) for idx, _input in enumerate(inputs)]
                else:
                    preds, sleep_len = backoff_response(inputs, sleep_len, args.max_len, model, use_chat_model)
                # iterate through each prediction in batch
                for item, pred_dict in zip(batch, preds):
                    tgt = item["tgt"]
                    if use_chat_model:
                        pred = pred_dict["message"]["content"]
                    elif args.llama:
                        pred = pred_dict
                    else:
                        pred = pred_dict["text"]
                    # Write prediction to file
                    preds_file.write(pred.replace("\n", " ") +"\n")
                    total += 1
                    # Evaluate whether prediction is correct
                    if args.task == "hans":
                        correct += int(eval_hans(pred, tgt))
                        correct_grouped[item["heuristic"]][str(item["tgt"])] += int(eval_hans(pred, tgt))
                        total_grouped[item["heuristic"]][str(item["tgt"])] += 1
                        if args.print_errors and not eval_hans(pred, tgt):
                            print(pred)
                    elif args.task == "transformations":
                        correct += int(eval_transformation(pred, tgt))
                        if args.print_errors and not eval_transformation(pred, tgt):
                            print(pred)
                    elif args.task == "passivization":
                        correct += int(eval_transformation(pred, tgt, transformation_type="passivization"))
                        if args.print_errors and not eval_transformation(pred, tgt, transformation_type="passivization"):
                            print(pred)
                    elif args.task == "tense":
                        correct += int(eval_tense(pred, tgt, item["diff_idx"]))
                        if args.print_errors and not eval_tense(pred, tgt, item["diff_idx"]):
                            print(pred)
        
    # Display overall accuracies
    print(f"Accuracy: {correct / total}")

    # If HANS, break down accuracies by heuristic type and label
    if args.task == "hans":
        for heuristic in correct_grouped:
            for entailment in correct_grouped[heuristic]:
                acc = correct_grouped[heuristic][entailment] / total_grouped[heuristic][entailment]
                print(f"Accuracy ({heuristic}) -- {entailment}: {acc}")
