import re
import pandas as pd
import openai
from datasets import load_dataset
from huggingface_hub import login


def passing():
    login('HF_TOKEN_REDACTED')
    openai.api_key = 'OTHER_API_KEY_REDACTED'
    openai.api_base = 'https://api.avalapis.ir/v1'

prompt = f'''
یک پرسش پزشکی چند گزینه ای انگلیسی به شما داده شده است وظیفه شما این است که این پرسش را به فارسی ترجمه کنید لطفا در فرمت زیر و بدون ارائه توضیحات اضافی پاسخ دهید.
# question
Question: {{question}}
Option A: {{optiona}}
Option B: {{optionb}}
Option C: {{optionc}}
Option D: {{optiond}}
Expression: {{expression}}

# format
Question:
Option A:
Option B:
Option C:
Option D:
Expression:
'''


def get_qa_elements(text):
    text = text + '\n'
    pattern1 = r"Question: (.*?)\nOption A: (.*?)\nOption B: (.*?)\nOption C: (.*?)\nOption D: (.*?)\nExpression: (.*?)\n"
    pattern2 = r"question: (.*?)\noption a: (.*?)\noption b: (.*?)\noption c: (.*?)\noption d: (.*?)\nexpression: (.*?)\n"
    match1 = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
    match2 = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
    if match1:
       question = match1.group(1).strip()
       option_a = match1.group(2).strip()
       option_b = match1.group(3).strip()
       option_c = match1.group(4).strip()
       option_d = match1.group(5).strip()
       expression = match1.group(6).strip()
    elif match2:
       question = match2.group(1).strip()
       option_a = match2.group(2).strip()
       option_b = match2.group(3).strip()
       option_c = match2.group(4).strip()
       option_d = match2.group(5).strip()
       expression = match2.group(6).strip()
    else:
       print('not matched input manually...  ' + '\n')
       print(text + '\n')
       question = input('question: ')
       print()
       option_a = input('option_a: ')
       print()
       option_b = input('option_b: ')
       print()
       option_c = input('option_c: ')
       print()
       option_d = input('option_d: ')
       print()
       expression = input('expression: ')
       print()
       print('$$$$$$$$$')
    return question,option_a,option_b,option_c,option_d,expression

def append_record_to_excel(file_path,id,question,op1,op2,op3,op4,expression,
                           correct_option,choice_type,subject_name,
                           topic_name):
    if question == '.':
       pass
    else:
        new_record = {
            'id': id,
            'Question': question,
            'Option1': op1,
            'Option2': op2,
            'Option3': op3,
            'Option4': op4,
            'Expression': expression,
            'Correct_answer': correct_option,
            'choice_type':choice_type,
            'subject_name':subject_name,
            'topic_name':topic_name
        }
        new_record_df = pd.DataFrame([new_record])
        try:
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        except FileNotFoundError:
            updated_df = new_record_df

        updated_df.to_excel(file_path, index=False)

def get_deepseek_answer(prompt):
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature = 0.0
    )
    result = response['choices'][0]['message']['content'].strip().lower()
    return result

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    passing()
    dataset = load_dataset('parquet', data_files='./train-00000-of-00001.parquet',split='train')
    i = 0
    bgn = 15078
    for j in range(182822):
        record = dataset[j]
        if i <= bgn - 1:
            i += 1
            continue
        id = record['id']
        question = record['question']
        option1 = record['opa']
        option2 = record['opb']
        option3 = record['opc']
        option4 = record['opd']
        cop = record['cop']
        choice_type = record['choice_type']
        exp = record['exp']
        subject_name = record['subject_name']
        topic_name = record['topic_name']
        run = True
        while run:
            try:
                translated_qa = get_deepseek_answer(
                prompt.format(question=question, optiona=option1, optionb=option2, optionc=option3, optiond=option4,
                            expression=exp))
                gquestion, goption_a, goption_b, goption_c, goption_d, gexpression = get_qa_elements(translated_qa)
                append_record_to_excel(file_path='./translated_qas.xlsx', id=id, question=gquestion,
                                       op1=goption_a, op2=goption_b, op3=goption_c, op4=goption_d,
                                       expression=gexpression, correct_option=cop, choice_type=choice_type,
                                       subject_name=subject_name, topic_name=topic_name)
                run = False
                i += 1
                print(i)
                print('==================================================================================')
            except:
                print('some error occurred')

