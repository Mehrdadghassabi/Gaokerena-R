import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import openai
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
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--file_no', type=int, required=True)
    return parser.parse_args()

args = parse_args()
token = 'HF_TOKEN_REDACTED'
login(token=token)

openai.api_key = 'OPENAI_API_KEY_REDACTED' #'OTHER_API_KEY_REDACTED'
# comment the below line if you use original OPENAI api
openai.api_base = 'https://api.deepseek.com'

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

prompt_extraction = f'''
an answer to a multiple choice question is given to you, your task is
to extract the option that has been chosen,
Please provide the extracted answer in the given format,without any additional explanation
# Format
[answer] e.g. [B]
# Answer
{{answer}}
'''

prompt_critic = f'''
یک پرسش چند گزینه ای پزشکی به همراه پاسخی که یک دانشجو به آن داده است به شما داده شده است
به کمک پاسخ صحیح پرسش که به همراه توضیحاتی پیوست شده است اشکالات پاسخ دانشجو را در فرمت گفته شده به او بگویید بدون آنکه مستقیما اشاره ای به گزینه صحیح بکنید.
در ضمن از توصیه به خواندن منابع نیز بپرهیزید و هر آنچه لازم است را به صورت مستقیم بنویسید.
# پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

## پاسخ صحیح
{{correct_answer}} \n

## توضیحات پاسخ صحیح
{{Expression}} \n

## پاسخ دانشجو
{{student_answer}} \n

#فرمت
1. نقد نخست
2. نقدم دوم
etc...
'''

prompt_refinement = f'''
گزینه ای که انتخاب کردید اشتباه بود با توجه به نقدی که توسط یک متخصص پزشکی برای شما فراهم شده است
دوباره به این پرسش قدم به قدم فکر کنید و زنجیره افکار (chain of thought) خود برای رسیدن به پاسخ را به طور کامل شرح دهید و  گزینه ای دیگر را انتخاب کنید.
همچنین تنها زنجیره افکار خود و پاسخ به پرسش مطرح شده را به عنوان خروجی برگرداندید و از به کار بردن عبارتی همچون "با توجه به نقد ارائه شده" یا "پاسخ اصلاح شده" بپرهیزید

## نقد
{{critic}} \n
'''

prompt_correction = f'''
یک پرسش پزشکی به همراه پاسخ یک دانشجو و پاسخ صحیح پرسش به شما داده شده است زنجیره افکار دانشجو را آنجایی که اشتباه است تصحیح کرده و به پاسخ مناسب برسید
در ضمن از به کار بردن عبارتی که به خواننده نشان دهد شما در حال تصحیح پاسخ دانشجو هستید بپرهیزید.
# پرسش
{{question}} \n

## گزینه ها
{{choices}} \n

## پاسخ صحیح
{{correct_answer}} \n

## توضیحات پاسخ صحیح
{{Expression}} \n

## پاسخ دانشجو
{{student_answer}} \n

'''


# @title utility
def extract_samples(task, numShot, model_prompt):
    questions, answer_choices, correct_answers = task_load(task, 'train')
    example_indexes = random.sample(range(len(questions)), numShot)
    example_list = []
    for i in example_indexes:
        example_list.append(model_prompt.format(question=questions[i], choices=format_choices(answer_choices[i]), answer=correct_answers[i]))
    return example_list

def get_ds_from_df(df,task):
    if 'filtered_qas' in task:
       df['Question'] = df['Question'].astype(str)
       df['Option1'] = df['Option1'].astype(str)
       df['Option2'] = df['Option2'].astype(str)
       df['Option3'] = df['Option3'].astype(str)
       df['Option4'] = df['Option4'].astype(str)
       df['Expression'] = df['Expression'].astype(str)
       df['Correct_answer'] = df['Correct_answer'].astype(str)
       df['choice_type'] = df['choice_type'].astype(str)
       df['subject_name'] = df['subject_name'].astype(str)
       df['topic_name'] = df['topic_name'].astype(str)
       ds = Dataset.from_pandas(df)
       return ds
    else:
       raise Exception("TASK NOT FOUND")

def append_record_to_excel(file_path,model_prompt,correct_answer, prefered_answer, rejected_answer, id):
    new_record = {
        'id': id,
        'prompt':model_prompt,
        'correct_choice':correct_answer,
        'prefered_answer': prefered_answer,
        'rejected_answer': rejected_answer
    }
    new_record_df = pd.DataFrame([new_record])
    try:
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = new_record_df

    updated_df.to_excel(file_path, index=False)

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)

def format_examples(examples):
    formatted_examples = []
    for row in examples:
        example = f'## Question {row["question"]} \n ## Answer {row["answer"]}'
        formatted_examples.append(example)
    return "\n".join(formatted_examples)

def task_load(task, split):
    if 'filtered_qas' in task:
        df = pd.read_excel('./medmcqa_translatio_filterd/' + task+'.xlsx')
        ds =get_ds_from_df(df,task)
        questions = [ds[i]['Question'] for i in range(len(ds))]
        answer_choices = [{"A": ds[i]['Option1'], "B": ds[i]['Option2'], "C": ds[i]['Option3'], "D": ds[i]['Option4']} for i in range(len(ds))]
        correct_answers = [chr(int(ds[i]['Correct_answer'])+65) for i in range(len(ds))]
        expressions = [ds[i]['Expression'] for i in range(len(ds))]
        ids = [ds[i]['id'] for i in range(len(ds))]
        return questions, answer_choices, correct_answers,expressions,ids
    else:
        raise Exception("TASK NOT FOUND")

def filterContext(context):
    end_tag = "</end>"
    if end_tag in context:
        return context.split(end_tag)[0] + end_tag
    return context

def run_inference(messages, engine, temp=1, max_tokens_output=200, tokenizer=None, model=None, local=False):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
         outputs = model.generate(inputs, max_new_tokens=max_tokens_output, do_sample = True, temperature=temp)
         text = tokenizer.batch_decode(outputs)[0]
         answer = re.split(r'<\|CHATBOT_TOKEN\|>', text)[-1].strip().removesuffix('<|END_OF_TURN_TOKEN|>').strip()
         return answer

def get_unlocal_AI_answer(prompt,m):
    response = openai.ChatCompletion.create(
        model=m,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature = 0.0
    )
    result = response['choices'][0]['message']['content'].strip().lower()
    return result

def extract_choice(answer):
    extr = get_unlocal_AI_answer(prompt_extraction.format(answer=answer),'deepseek-chat')
    if extr == '[a]' or  extr == '[A]' or extr == 'a' or extr == 'A' or extr == '[الف]' or extr == 'الف' or extr == '[آ]' or extr == 'آ' or extr == '[ا]' or extr == 'ا':
       return 'A'
    elif extr == '[b]' or extr == '[B]' or extr == 'b' or extr == 'B' or extr == '[ب]' or extr == 'ب':
       return 'B'
    elif extr == '[c]' or extr == '[C]' or extr == 'c' or extr == 'C' or extr == '[ج]' or extr == 'ج':
       return 'C'
    elif extr == '[d]' or extr == '[D]' or extr == 'd' or extr == 'D' or extr == '[د]' or extr == 'د':
       return 'D'
    else:
       return 'invalid'

def resume_the_teaching_with_hint(question_list, answer_choices_list, correct_answer_list,expression_list,bgn, end):
    question_list = question_list[bgn:end]
    answer_choices_list = answer_choices_list[bgn:end]
    correct_answer_list = correct_answer_list[bgn:end]
    expression_list = expression_list[bgn:end]
    for i, (question, answer_choices, correct_answer,expression) in tqdm(enumerate(zip(question_list, answer_choices_list,correct_answer_list,expression_list))):
        try: 
            prompt = prompt_eval_bare_COT
            model_prompt = prompt.format(question=question, choices=format_choices(answer_choices))
            messages = [{"role": "user", "content": f"{model_prompt}"}]
            student_answer = run_inference(messages, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, local=True)
            ext = extract_choice(student_answer)
            #print(student_answer)
            #print(ext)
            #print(correct_answer)
            #print('-----------------------------------------------')
            if ext == correct_answer:
                continue
            else:
                rejected_answer = student_answer
                messages.append({"role": "assistant", "content": student_answer})
                count = 0
                while count < 10:
                        count += 1
                        p = prompt_critic.format(question=question,
                                                choices=format_choices(answer_choices),
                                                correct_answer=correct_answer,
                                                Expression=expression,
                                                student_answer=student_answer)
                        critic = get_unlocal_AI_answer(p,'deepseek-reasoner')
                        messages.append({"role": "user", "content":prompt_refinement.format(critic=critic)})
                        student_answer = run_inference(messages, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, local=True)
                        messages.append({"role": "assistant", "content": student_answer})
                        ext = extract_choice(student_answer)
                        if ext == correct_answer:
                            prefered_answer = student_answer
                            file_path = 'prefered_rejected_pairs.xlsx'
                            append_record_to_excel(file_path,model_prompt,correct_answer, prefered_answer, rejected_answer, id)
                            #print(student_answer)
                            #print(ext)
                            #print(correct_answer)
                            #print('-----------------------------------------------')
                            continue

                        #print(student_answer)
                        #print(ext)
                        #print(correct_answer)
                        #print('-----------------------------------------------')
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")


def resume_teaching(question_list, answer_choices_list, correct_answer_list,expression_list, bgn, end):
    question_list = question_list[bgn:end]
    answer_choices_list = answer_choices_list[bgn:end]
    correct_answer_list = correct_answer_list[bgn:end]
    expression_list = expression_list[bgn:end]
    for i, (question, answer_choices, correct_answer,expression,id) in tqdm(enumerate(zip(question_list, answer_choices_list,correct_answer_list,expression_list,ids))):
        try:
            prompt = prompt_eval_bare_COT
            model_prompt = prompt.format(question=question, choices=format_choices(answer_choices))
            messages = [{"role": "user", "content": f"{model_prompt}"}]
            student_answer = run_inference(messages, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, local=True)
            ext = extract_choice(student_answer)
            if ext == correct_answer:
                continue
            else:
                rejected_answer = student_answer
                messages.append({"role": "assistant", "content": student_answer})
                p = prompt_correction.format(question=question, choices=format_choices(answer_choices),
                                                correct_answer=correct_answer,Expression=expression,student_answer=student_answer)
                prefered_answer =  get_unlocal_AI_answer(p,'deepseek-reasoner')
                file_path = f'prefered_rejected_pairs_{TASK_PART}_{args.file_no}.xlsx'
                append_record_to_excel(file_path,model_prompt,correct_answer, prefered_answer, rejected_answer, id)
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")


# @title model setting

print("RUNNING NORMAL IMPLEMENTATION")
ENGINE = "CohereForAI/aya-expanse-8b"
SPLIT = "test"
ENGINE_TEMPERATURE = 1.0
MAX_TOKEN_OUTPUT = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(ENGINE)
model = AutoModelForCausalLM.from_pretrained(ENGINE,torch_dtype=torch.bfloat16).to(device)
model.eval()
## OUTPUT RUN INFO:
print("Model Running: " + ENGINE)

# @title Load the test


TASK_PART = 'part5'
TASK = 'filtered_qas_' + TASK_PART
question_list, answer_choices_list, correct_answer_list,expression_list,ids = task_load(TASK, SPLIT)
print(f"{TASK} loaded succesfully. Now conducting evaluation on {len(question_list)} samples.")

resume_teaching(question_list, answer_choices_list, correct_answer_list,expression_list,bgn = args.begin, end=args.end)