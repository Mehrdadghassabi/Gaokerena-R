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
import math
from collections import Counter
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    # parser.add_argument("--exp_no", type=int, required=True)
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
prompt_get_part = f'''

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
'''

prompt_eval = f'''
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
این یک پرسش پزشکی است به همراه گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n
'''

prompt_eval_bare = f'''
به پرسش زیر به پاسخ بدهید و تنها از زبان فارسی استفاده کنید!
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

\n
'''

prompt_eval_bare_COT = f'''
به پرسش زیر قدم به قدم فکر کنید و زنجیره افکار (chain of thought) خود برای رسیدن به پاسخ را به طور کامل شرح دهید و تنها یک گزینه را انتخاب کنید.
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
'''

prompt_eval_bare_fully = f'''
{{question}} \n
{{choices}}
'''

prompt_eval_bare_fully_with_examples = f'''
{{examples}} \n
{{question}} \n
{{choices}}
'''

prompt_eval_with_examples = f'''
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
'''

prompt_eval_context_bare = f'''
{{context}} \n
{{question}} \n
{{choices}}
'''
prompt_eval_with_context = f'''
## Context
{{context}} \n

## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n
این یک متن از یک کتاب مرجع است به همراه یک پرسش پزشکی و گزینه هایی که میتوانید انتخاب کنید. به این پرسش قدم به قدم فکر کرده و پاسخ خود را به صورت زیر بیان کنید:
پاسخ تولید شده توسط مدل که قدم به قدم به آن فکر شده. بنابراین پاسخ نهایی [پاسخ نهایی مدل (e.g. A,B,C,D)] میباشد. به عنوان مثال [A]
\n '''

prompt_eval_with_context_and_examples = f'''
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
\n '''

prompt_example = f'''
## پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

## پاسخ
{{answer}} \n
'''

prompt_check_incorrect_information = f'''
پنج متن پزشکی به زبان فارسی به شما داده شده است. کدام پاسخ کمترین اطلاعات غلط راارائه میدهد؟

## متن شماره یک
{{passage1}} \n

## متن شماره دو
{{passage2}} \n

## متن شماره سه
{{passage3}} \n

## متن شماره چهار
{{passage4}} \n

## متن شماره پنج
{{passage5}} \n

\n
'''



# @title utility
# Set openai key if using gpt4o as engine.
#os.environ['OPENAI_API_KEY'] = "OPEN AI KEY HERE"
def get_ds_from_df(df):
       df['Question'] = df['Question'].astype(str)
       df['question_choices'] = df['question_choices'].astype(str)
       df['correct_answer'] = df['correct_answer'].astype(str)
       df['model_prompt'] = df['model_prompt'].astype(str)
       df['AI_answer'] = df['AI_answer'].astype(str)
       df['AI_chosen_answer'] = df['AI_chosen_answer'].astype(str)
       ds = Dataset.from_pandas(df)
       return ds

def append_record_to_excel(file_path,question,
                           correct_answer,prompt,guide_answer):
    new_record = {
        'Question': question,
        'correct_answer': correct_answer,
        'model_prompt':  prompt,
        'guide_answer': guide_answer
    }
    new_record_df = pd.DataFrame([new_record])
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        else:
            updated_df = new_record_df
    except Exception as e:
        print(f"⚠️ Warning: Could not read {file_path} ({e}). Recreating file.")
        updated_df = new_record_df

    updated_df.to_excel(file_path, index=False)

def load_answers(base_path):
    df1 = pd.read_excel(f'{base_path}/gao_experiment_no1.xlsx')
    df2 = pd.read_excel(f'{base_path}/gao_experiment_no2.xlsx')
    df3 = pd.read_excel(f'{base_path}/gao_experiment_no3.xlsx')
    df4 = pd.read_excel(f'{base_path}/gao_experiment_no4.xlsx')
    df5 = pd.read_excel(f'{base_path}/gao_experiment_no5.xlsx')

    ds1 =get_ds_from_df(df1)
    ds2 =get_ds_from_df(df2)
    ds3 =get_ds_from_df(df3)
    ds4 =get_ds_from_df(df4)
    ds5 =get_ds_from_df(df5)

    questions = [[ds1[i]['Question'] for i in range(len(ds1))],
                 [ds2[i]['Question'] for i in range(len(ds2))],
                 [ds3[i]['Question'] for i in range(len(ds3))],
                 [ds4[i]['Question'] for i in range(len(ds4))],
                 [ds5[i]['Question'] for i in range(len(ds5))]]

    question_choices = [[ds1[i]['question_choices'] for i in range(len(ds1))],
                        [ds2[i]['question_choices'] for i in range(len(ds2))],
                        [ds3[i]['question_choices'] for i in range(len(ds3))],
                        [ds4[i]['question_choices'] for i in range(len(ds4))],
                        [ds5[i]['question_choices'] for i in range(len(ds5))]]

    correct_answers = [[ds1[i]['correct_answer'] for i in range(len(ds1))],
                       [ds2[i]['correct_answer'] for i in range(len(ds2))],
                       [ds3[i]['correct_answer'] for i in range(len(ds3))],
                       [ds4[i]['correct_answer'] for i in range(len(ds4))],
                       [ds5[i]['correct_answer'] for i in range(len(ds5))]]

    model_prompts = [[ds1[i]['model_prompt'] for i in range(len(ds1))],
                     [ds2[i]['model_prompt'] for i in range(len(ds2))],
                     [ds3[i]['model_prompt'] for i in range(len(ds3))],
                     [ds4[i]['model_prompt'] for i in range(len(ds4))],
                     [ds5[i]['model_prompt'] for i in range(len(ds5))]]

    AI_answers = [[ds1[i]['AI_answer'] for i in range(len(ds1))],
                  [ds2[i]['AI_answer'] for i in range(len(ds2))],
                  [ds3[i]['AI_answer'] for i in range(len(ds3))],
                  [ds4[i]['AI_answer'] for i in range(len(ds4))],
                  [ds5[i]['AI_answer'] for i in range(len(ds5))]]

    AI_chosen_answers = [[ds1[i]['AI_chosen_answer'] for i in range(len(ds1))],
                         [ds2[i]['AI_chosen_answer'] for i in range(len(ds2))],
                         [ds3[i]['AI_chosen_answer'] for i in range(len(ds3))],
                         [ds4[i]['AI_chosen_answer'] for i in range(len(ds4))],
                         [ds5[i]['AI_chosen_answer'] for i in range(len(ds5))]]
    return questions, question_choices,correct_answers,model_prompts, AI_answers, AI_chosen_answers

def resume_the_test(questions, question_choices,
     correct_answers, model_prompts, AI_answers, AI_chosen_answers, file_path, bgn):
    questions = list(zip(*questions[bgn:][:]))
    question_choices = list(zip(*question_choices[bgn:][:]))
    correct_answers = list(zip(*correct_answers[bgn:][:]))
    model_prompts = list(zip(*model_prompts[bgn:][:]))
    AI_answers = list(zip(*AI_answers[bgn:][:]))
    AI_chosen_answers = list(zip(*AI_chosen_answers[bgn:][:]))
    # print("Heeey 0")
    # print(questions)

    for i, (question, question_choice,correct_answer, model_prompt,
            AI_answer, AI_chosen_answer) in tqdm(enumerate(zip(questions, question_choices,
            correct_answers, model_prompts, AI_answers, AI_chosen_answers))):
            # print("Heeey 1")
            answers = [AI_chosen_answer[0],AI_chosen_answer[1],AI_chosen_answer[2],AI_chosen_answer[3],AI_chosen_answer[4]]
            answer_counts = Counter(answers)


            for option, count in answer_counts.items():
                # print(count)
                if count >= 3:
                   model_prmpt = ''
                   guide_answer = option
                   break
                else:
                    prompt = prompt_check_incorrect_information
                    model_prmpt = prompt.format(passage1=AI_answer[0],passage2=AI_answer[1],passage3=AI_answer[2],passage4=AI_answer[3],passage5=AI_answer[4])
                    guide_answer = run_inference(model_prmpt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, local=True)
                    # print("Done!!")

            append_record_to_excel(file_path, question[0],
                           correct_answer[0],model_prmpt,guide_answer)
            # print("Appended!!")

            if i == STOP_GEN-1:
                break

def run_inference(content, engine, temp=1, max_tokens_output=200, tokenizer=None, model=None, local=False):
    if local:
        messages = [{"role": "user", "content": f"{content}"}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda:0')
        with torch.no_grad():
             outputs = model.generate(inputs, max_new_tokens=max_tokens_output, do_sample = True, temperature=temp)
             text = tokenizer.batch_decode(outputs)[0]
             answer = re.sub(r'<\|END_OF_TURN_TOKEN\|>$', '', text.split("model")[-1].split("<|CHATBOT_TOKEN|>")[1])
             return answer
    # else:
    #     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    #     messages = [{"role": "user", "content": f"{content}"}]
    #     response = client.chat.completions.create(
    #         model=engine,
    #         messages=messages,
    #         temperature=temp,
    #         max_tokens=max_tokens_output,
    #         frequency_penalty=0.0
    #     )
    #     response_text = response.choices[0].message.content
    #     return response_text



# @title model setting
print("RUNNING NORMAL IMPLEMENTATION")
ENGINE = "CohereForAI/aya-expanse-8b"
SPLIT = "test"
ENGINE_TEMPERATURE = 1
MAX_TOKEN_OUTPUT = 1024
NSHOT = 0
STOP_GEN = 10000000 ## For testing purposes; stop generating after {STOP_GEN} amount of test-questions
TASK = args.task # Options ["kopp", 'mmlu-anatomy', 'mmlu-professional_medicine', 'mmlu-college_biology', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-medical_genetics']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    ENGINE,
    torch_dtype=dtype,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")
model.eval()
## OUTPUT RUN INFO:
print("Model Running: " + ENGINE)

# @title Load the test
questions, question_choices,correct_answers, model_prompts,AI_answers, AI_chosen_answers = load_answers(base_path=f'./evaluations/zeroshot-COT/{task_to_dir[TASK]}/gaokerena-r1.0/')

base_path = f'./evaluations/comparison_exp/'
out_path = f'{base_path}/aya_trained_{task_to_dir[TASK]}.xlsx'

# Since google colab usage time is limited & this test takes days to complete
#  we need to concatenate the result of many session to get the final result
#   so set the bgn variable to number of question that has been solved in previous sessions
resume_the_test(questions, question_choices,
     correct_answers, model_prompts, AI_answers, AI_chosen_answers, file_path=out_path, bgn = 0)