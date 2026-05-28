API = 'yours' # gemini 2.5-pro
# API = 'yours' #xinnan
# API = 'yours'
from google import genai
import json

# The client gets the API key from the environment variable `GEMINI_API_KEY`.


model = 'qwen3_14b'
dataset = '2wiki'
# model = 'qwen3_14b'
# dataset = 'manu_musique'
file_name = f'./data_{dataset}/{model}_answer.json'
with open(file_name,'r') as f:
    data_json = json.load(f)


if dataset == 'gsm8k':
    client = genai.Client(api_key=API)

    response = client.models.generate_content(
        model="gemini-2.5-pro", contents="Explain how AI works in a few words"
    )
    print(response.text)
elif dataset == 'WebQSP' or dataset == 'WebQSP_ROG' or dataset == 'RAMDOCS' or dataset == 'hotpotqa' or dataset == '2wiki' or dataset == 'musique' or dataset == 'manu_musique' or dataset == 'CWQ_ROG':
    client = genai.Client(api_key=API)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents="Explain how AI works in a few words"
    )
    print(response.text)

new_data = []
judges = []
from tqdm import tqdm
for idx,data in tqdm(enumerate(data_json), total=len(data_json)):
    question, gold_ans, ans = data['question'], data['gold_ans'], data['ans']
    if dataset == 'WebQSP_ROG':
        path = data['path']
    if dataset == 'CWQ_ROG':
        path = data['path']
    if dataset == 'hotpotqa':
        path = data['path']
    if dataset == '2wiki':
        path = data['path']
    if dataset == 'musique':
        path = data['path']
    if dataset == 'manu_musique':
        path = data['path']
        q_type = data['type']
    if dataset == 'RAMDOCS':
        path = data['path']
        group = data['group']
    if dataset == 'gsm8k':
        response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
        )
        judge = response.text
        new_data.append(dicts)
        print(judge)
    elif dataset == 'WebQSP'  or dataset == 'CWQ_ROG' or dataset == 'WebQSP_ROG' or dataset == 'RAMDOCS' or dataset == 'hotpotqa' or dataset == '2wiki' or dataset == 'musique' or dataset == 'manu_musique':
        # prompt = f'Given the question from {dataset}, {question} and a ground truth answer {gold_ans}, the predict answer is {ans}. Is the predict answer right? only select Yes or No.'
        # prompt = f'Given dataset {dataset}, question {question}, ground truth {gold_ans}, and prediction {ans}, answer Yes only if the prediction is identical to the ground truth; otherwise answer No. Output only Yes or No.'
        # prompt = f'Given dataset {dataset}, question {question}, ground truth answers {gold_ans}, and prediction {ans}, answer "Yes" only if the prediction MATCHES ALL ground truths under a LENIENT rule: ignore case, trim and collapse whitespace, ignore surrounding quotes, and ignore trailing sentence-ending punctuation (.,!? ...). For a list prediction, each ground truth must have a corresponding matched item; for a single-string prediction, it must cover all ground truths as tokens/items regardless of order. Extra neutral context is allowed, but missing any ground truth yields "No". Output only Yes or No.'
        prompt = f'Given the question from {dataset}, {question} and a ground truth answer {gold_ans}, the predict answer is {ans}. Is the prediction COVERS ALL gold_ans? only select Yes or No.' #all
        # prompt = (
        #     f"The gold answers are: {gold_ans}. "
        #     f"The predicted answer is: {ans}. "
        #     "If the predicted answer contains at least ONE of the gold answers, reply Yes. "
        #     "Otherwise reply No. Only output Yes or No."
        # )


        for _retry in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite", contents=prompt
                )
                judge = response.text
                break
            except Exception as e:
                import time
                print(f"\nAPI error (attempt {_retry+1}/5): {e}")
                time.sleep(5 * (_retry + 1))  # incremental wait: 5s, 10s, 15s, 20s, 25s
        else:
            judge = "No"  # retries exhausted, default to No
        judges.append(judge)
        print(judge)
    elif dataset =='MAWPS':
        # print(gold_ans)
        if str(gold_ans) in ans:
            judge = True
        else:
            judge = False
        judges.append(judge)
    dicts = {
        "question": question,
        "path": path,
        "gold_ans": gold_ans,
        "ans": ans,
        "judge": judge
    }
    if dataset == 'manu_musique':
        dicts["type"] = q_type
    # if idx ==10:break
    new_data.append(dicts)
# print(sum(judges)/len(judges))
yes_count = judges.count('Yes') + judges.count('yes')
print(f"Accuracy: {yes_count/len(judges):.2%} ({yes_count}/{len(judges)})")
with open(file_name,'w') as f:
    json.dump(new_data, f)
