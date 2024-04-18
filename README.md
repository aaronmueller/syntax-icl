# In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax
This repository contains materials for the paper
"In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax" 
(Mueller et al., 2024). This paper was presented and published at NAACL 2024.
A preprint is available [on arXiv](https://arxiv.org/abs/2311.07811).

If you find the code or data useful in your research, please use the citation at the end of the README.


## Prompts and Evaluation Data
Prompts and evaluation datasets can be found in the [`data`](data) folder.

The directories immediately under `data`
are split by task: `question_formation`, `tense_reinflection`, and `hans`. Within each of these folders are up to two
evaluation datasets (as .json files): `test.id.json` (in-distribution evaluation data), `test.ood.json` (out-of-distribution
evaluation data), or, in the case of HANS, simply `test.json`.

Each of these folders also contains prompts: `no_cot.txt` corresponds to no chain-of-thought in the prompt;
`verbal_cot.txt` incorporates meta-linguistic information about the inputs, explained verbally; and `code_cot.txt`
incorporates meta-linguistic information about the inputs in a Python code format.


## Generation and Evaluation Scripts
The [`generate_and_eval.py`](generate_and_eval.py) script contains code for running generation and evaluating
model predictions. This script relies on evaluation logic in `evaluation.py`, as well as
generation helper functions in `generation_utils.py`.

As an example, to evaluate GPT-3.5 `text-davinci-003` on the OOD question formation data with no chain-of-thought prompting,
run the following command from the root directory of this repository:
```
python generate_and_eval.py question_formation no_cot.txt --oai_model text-davinci-003 --max_length 64
```

And to evaluate Llama 2 (70B) on in-distribution question formation data:
```
python generate_and_eval.py question_formation no_cot.txt --llama meta-llama/Llama-2-70b-hf --max_length 64 -t test.id.json
```

Please note that to run the OpenAI models, you will need to set the environment variable `$OPENAI_API_KEY` to
your API key. For security reasons, **do not** commit this key anywhere in your code.


## Evaluating Reasoning Accuracy and Faithfulness
There are scripts for detailed evaluations of reasoning accuracy (the accuracy of the model's chain-of-thought reasoning) and faithfulness (whether the model's final answer agrees with its own chain-of-thought reasoning). These are provided under `data/question_formation/detailed_eval.py` and `data/tense_reinflection/detailed_eval.py`.

Note that these scripts will only work for generations using the Code CoT format.


## Citation
```
@inproceedings{mueller-etal-icl-2024,
    title = {In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax},
    author = {Mueller, Aaron and Webson, Albert and Petty, Jackson and Linzen, Tal},
    year = "2024",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics"
}
```

## License
The contents of this repository are licensed under an MIT license.