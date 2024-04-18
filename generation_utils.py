import openai

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
    return response["choices"]


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


def load_llama(model_name):
    import torch

    if model_name.startswith("meta-llama/Llama-2"):
        from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
        llama_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(    # use 4-bit quantization to make it fit on a single GPU
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        llama_model = LlamaForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    elif model_name.startswith("codellama/"):
        from transformers import CodeLlamaTokenizer, AutoModelForCausalLM
        llama_tokenizer = CodeLlamaTokenizer.from_pretrained(model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(     # load in half-precision
            model_name,
            torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        raise ValueError("Unsupported model: ")

    return llama_tokenizer, llama_model


def get_llama_response(tokenizer, model, prompts, max_len):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids,
                            top_p=0.9,
                            temperature=0.1,
                            max_new_tokens=max_len,
                            pad_token_id=tokenizer.eos_token_id)
    strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return strings