# from openai import OpenAI
import os
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from huggingface_hub import login
import torch
import random
import re
import numpy as np
from collections import Counter
from peft.peft_model import PeftModel
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--exp_no", type=int, required=True)
    return parser.parse_args()


args = parse_args()

task_to_dir = {
    "kopp": "kopp",
    "mmlu-anatomy": "MMLU_Anatomy",
    "mmlu-professional_medicine": "MMLU-professional_medicine",
    "mmlu-college_biology": "MMLU-college_biology",
    "mmlu-college_medicine": "MMLU-college_medicine",
    "mmlu-clinical_knowledge": "MMLU-clinical_knowledge",
    "mmlu-medical_genetics": "MMLU-medical_genetics",
}

# @title prompts
prompt_get_part = f"""

## پرسش
{{question}} \n
این یک پرسش پزشکی است برای طبقه بندی پرسش یکی از موضوعات زیر را انتخاب کنید به گونه ای که بیشترین تطابق را با پرسش داشته باشد

Topic List = [
    "Part 1: The Profession of Medicine",
    "Part 2: Cardinal Manifestations and Presentation of Diseases",
    "Part 3: Pharmacology",
    "Part 4: Oncology and Hematology",
    "Part 5: Infectious Diseases",
    "Part 6: Disorders of the Cardiovascular System",
    "Part 7: Disorders of the Respiratory System",
    "Part 8: Critical Care Medicine",
    "Part 9: Disorders of the Kidney and Urinary Tract",
    "Part 10: Disorders of the Gastrointestinal System",
    "Part 11: Immune-Mediated, Inflammatory, and Rheumatologic Disorders",
    "Part 12: Endocrinology and Metabolism",
    "Part 13: Neurologic Disorders",
    "Part 14: Poisoning, Drug Overdose, and Envenomation",
    "Part 15: Disorders Associated with Environmental Exposures",
    "Part 16: Genes, the Environment, and Disease",
    "Part 17: Global Medicine",
    "Part 18: Aging",
    "Part 19: Consultative Medicine",
    "Part 20: Frontiers"
]
به عنوان پاسخ تنها و تنها یک عدد برگردانید به گونه ای که بیشترین تطابق را با پرسش داشته باشد.
"""

prompt_eval = f"""
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
این یک پرسش پزشکی است به همراه گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n
"""

prompt_eval_bare = f"""
به پرسش زیر به پاسخ بدهید و تنها از زبان فارسی استفاده کنید!
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

\n
"""

prompt_eval_bare_COT = f"""
به پرسش زیر قدم به قدم فکر کنید و زنجیره افکار (chain of thought) خود برای رسیدن به پاسخ را به طور کامل شرح دهید و تنها یک گزینه را انتخاب کنید.
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
"""

prompt_eval_bare_fully = f"""
{{question}} \n
{{choices}}
"""

prompt_eval_bare_fully_with_examples = f"""
{{examples}} \n
{{question}} \n
{{choices}}
"""

prompt_eval_with_examples = f"""
## نمونه ها
{{examples}}

Above are examples for medical Q&A.

## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

این یک پرسش پزشکی است به همراه گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n
"""

prompt_eval_context_bare = f"""
{{context}} \n
{{question}} \n
{{choices}}
"""
prompt_eval_with_context = f"""
## Context
{{context}} \n

## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
این یک متن از یک کتاب مرجع است به همراه یک پرسش پزشکی و گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n """

prompt_eval_with_context_and_examples = f"""
## نمونه ها
{{examples}}
در بالا نمونه هایی از پرسش پاسخ پزشکی آورده شده است.

## متن
{{context}} \n

## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

این یک پرسش پزشکی است به همراه گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n """

prompt_example = f"""
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

## پاسخ
{{answer}} \n
"""


# @title utility
# Set openai key if using gpt4o as engine.
# os.environ['OPENAI_API_KEY'] = "OPEN AI KEY HERE"
def calculate_entropy(answers):
    answer_counts = Counter(answers)
    total_answers = len(answers)
    probabilities = [count / total_answers for count in answer_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0) / np.log10(
        total_answers
    )

    return entropy


def extract_samples(task, numShot, model_prompt):
    questions, answer_choices, correct_answers = task_load(task, "train")
    example_indexes = random.sample(range(len(questions)), numShot)
    example_list = []
    for i in example_indexes:
        example_list.append(
            model_prompt.format(
                question=questions[i],
                choices=format_choices(answer_choices[i]),
                answer=correct_answers[i],
            )
        )
    return example_list


def get_ds_from_df(df, task):
    if task == "kopp":
        df["Question"] = df["Question"].astype(str)
        df["Option1"] = df["Option1"].astype(str)
        df["Option2"] = df["Option2"].astype(str)
        df["Option3"] = df["Option3"].astype(str)
        df["Option4"] = df["Option4"].astype(str)
        df["Topic"] = df["Topic"].astype(str)
        df["Source"] = df["Source"].astype(str)
        df["Correct answer"] = df["Correct answer"].astype(str)
        ds = Dataset.from_pandas(df)
        return ds
    elif "mmlu" in task:
        df["question"] = df["question"].astype(str)
        df["option1"] = df["option1"].astype(str)
        df["option2"] = df["option2"].astype(str)
        df["Option3"] = df["option3"].astype(str)
        df["option4"] = df["option4"].astype(str)
        df["answer"] = df["answer"].astype(str)
        ds = Dataset.from_pandas(df)
        return ds
    else:
        raise Exception("TASK NOT FOUND")


def resume_the_test(
    question_list, answer_choices_list, correct_answer_list, out_path, bgn
):
    question_list = question_list[bgn:]
    answer_choices_list = answer_choices_list[bgn:]
    correct_answer_list = correct_answer_list[bgn:]
    for i, (question, answer_choices, correct_answer) in tqdm(
        enumerate(zip(question_list, answer_choices_list, correct_answer_list))
    ):
        context = ""
        if NSHOT == 0:
            prompt = prompt_eval_bare_COT
        else:
            prompt = prompt_eval_bare_fully_with_examples

        if NSHOT != 0:
            examples = extract_samples(TASK, NSHOT, prompt_example)
            model_prompt = prompt.format(
                question=question,
                choices=format_choices(answer_choices),
                examples=("\n").join(examples),
                context=filterContext(context),
            )
        else:
            model_prompt = prompt.format(
                question=question,
                choices=format_choices(answer_choices),
                context=filterContext(context),
            )

        AI_answer = run_inference(
            model_prompt,
            ENGINE,
            ENGINE_TEMPERATURE,
            MAX_TOKEN_OUTPUT,
            tokenizer,
            model,
            local=True,
        )
        append_record_to_excel(
            out_path, question, answer_choices, correct_answer, model_prompt, AI_answer
        )

        if i == STOP_GEN - 1:
            break


def append_record_to_excel(
    out_path, question, question_choices, correct_answer, model_prompt, AI_answer
):
    new_record = {
        "Question": question,
        "question_choices": question_choices,
        "correct_answer": correct_answer,
        "model_prompt": model_prompt,
        "AI_answer": AI_answer,
    }
    new_record_df = pd.DataFrame([new_record])
    try:
        existing_df = pd.read_excel(out_path)
        updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = new_record_df

    updated_df.to_excel(out_path, index=False)


def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x, y in a:
        final_answers.append(f"[{x}] : {y}")
    return "\n".join(final_answers)


def format_examples(examples):
    formatted_examples = []
    for row in examples:
        example = f'## Question {row["question"]} \n ## Answer {row["answer"]}'
        formatted_examples.append(example)
    return "\n".join(formatted_examples)


def task_load(task, base_path, split):
    if  "kopp" in task:
        df = pd.read_excel(f"{base_path}/{task}.xlsx")
        ds = get_ds_from_df(df, task)
        questions = [ds[i]["Question"] for i in range(len(ds))]
        answer_choices = [
            {
                "A": ds[i]["Option1"],
                "B": ds[i]["Option2"],
                "C": ds[i]["Option3"],
                "D": ds[i]["Option4"],
            }
            for i in range(len(ds))
        ]
        correct_answers = [
            chr(int(ds[i]["Correct answer"]) + 64) for i in range(len(ds))
        ]
        return questions, answer_choices, correct_answers
    elif "mmlu" in task:
        df = pd.read_excel(f"{base_path}/{task}_fa.xlsx")
        ds = get_ds_from_df(df, task)
        questions = [ds[i]["question"] for i in range(len(ds))]
        answer_choices = [
            {
                "A": ds[i]["option1"],
                "B": ds[i]["option2"],
                "C": ds[i]["option3"],
                "D": ds[i]["option4"],
            }
            for i in range(len(ds))
        ]
        correct_answers = [chr(int(ds[i]["answer"]) + 64) for i in range(len(ds))]
        return questions, answer_choices, correct_answers
    else:
        raise Exception("TASK NOT FOUND")


def filterContext(context):
    end_tag = "</end>"
    if end_tag in context:
        return context.split(end_tag)[0] + end_tag
    return context


def run_inference(
    content,
    engine,
    temp=1,
    max_tokens_output=200,
    tokenizer=None,
    model=None,
    local=False,
):
    if local:
        messages = [{"role": "user", "content": f"{content}"}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens_output,
                do_sample=True,
                temperature=temp,
            )
            text = tokenizer.batch_decode(outputs)[0]
            answer = re.sub(
                r"<\|END_OF_TURN_TOKEN\|>$",
                "",
                text.split("model")[-1].split("<|CHATBOT_TOKEN|>")[1],
            )
            return answer
    # else:
    #     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    #     messages = [{"role": "user", "content": f"{content}"}]
    #     response = client.chat.completions.create(
    #         model=engine,
    #         messages=messages,
    #         temperature=temp,
    #         max_tokens=max_tokens_output,
    #         frequency_penalty=0.0,
    #     )
    #     response_text = response.choices[0].message.content
    #     return response_text


# @title model setting

print("RUNNING NORMAL IMPLEMENTATION")
ENGINE = "./Aya-Trained-16v16-0.05dp-GaoV/checkpoint-2479"
SPLIT = "test"
ENGINE_TEMPERATURE = 1
MAX_TOKEN_OUTPUT = 1024
TASK = args.task
EXP_NUM = args.exp_no
NSHOT = 0
STOP_GEN = 10000000  ## For testing purposes; stop generating after {STOP_GEN} amount of test-questions
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(ENGINE, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    ENGINE, torch_dtype=torch.bfloat16, local_files_only=True
).to(device)
model.eval()
## OUTPUT RUN INFO:
print("Model Running: " + ENGINE)
base_path = f'../evaluations/zeroshot-COT/{task_to_dir[TASK]}'
out_path = f'{base_path}/gaokerena-vr1.0/aya_trained_16v16_0.05dp_experiment_no_{EXP_NUM}.xlsx'
# @title Load the test
question_list, answer_choices_list, correct_answer_list = task_load(TASK, base_path, SPLIT)
print(
    f"{TASK} loaded succesfully. Now conducting evaluation on {len(question_list)} samples."
)


# Since google colab usage time is limited & this test takes days to complete
#  we need to concatenate the result of many session to get the final result
#   so set the bgn variable to number of question that has been solved in previous sessions
resume_the_test(
    question_list, answer_choices_list, correct_answer_list, out_path=out_path, bgn=0
)
