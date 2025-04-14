import os
import re
import gzip
import json
import time
import random
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


R2R_VALUNSEEN_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
DIR_NAME = "data/datasets/LLM_REPLYS_VAL_UNSEEN/"
FILE_NAME = "llm_reply_valunseen"
TEMP_SAVE_PATH = '/data/ckh/Zero-Shot-VLN-FusionMap-mp/tests/llm_reply_valunseen_temp.json'
dones = 0


prompt_template = f"""Parse a navigation instruction delimited by triple quotes and your task is to perform the following actions:
1. Extract Destination: Understand the entire instruction and summarize a description of the destination. The description should be a sentence containing landmark and roomtype. The description of the destination should not accurately describe the orientation and order. Here are examples about destination: "second room on the left" -> "room"(neglect order and direction); "between the bottom of the first stair and the console table in the entry way" -> "console table near entry way"(simplify description); "in front of the railing about halfway between the two upstairs rooms" -> "railing near two upstair rooms";
2. Split instructions: Split the instruction into a series of sub-instructions according to the execution steps. Each sub-instruction contain one landmark.
3. Infer agent's state constraints: Infer the state constraints that the agent should satisfy for each sub-instruction. There're thee constraint types: location constraints, diretion constraints, object constraints. You need to select an appropriate constraint type and give the corresponding constraint object. Direction constraint object has two types: left, right. Constraints can format as a tuple: (constraint type, constraint object)
4. Make a decision: Analyze the landmarks, actions, and directions in each sub-instruction to determine how the agent should act. For a landmark, the agent has three options: approach, move away, or approach and then move away. For direction, the agent has three options: turn left, turn right, or go forward
Provide your answer in JSON format with the following details:
1. use the following keys: destination, sub-instructions, state-constraints, decisions
2. the value of destination is a string
3. the value of sub-instructions is a list of all sub-instructions
4. the value of state-constraints is a JSON. The key is index start from zero and the value is a list of all constraints, each constraint is a tuple
5. the value of decisions is a nested JSON. The first level JSON's key is index start from zero and it;s value is second level JONS with keys: landmarks, directions. The value of landmarks is a list of tuples, each tuple contains (landmark, action). The value of directions is a list of direction choice for each sub-instruction.
An Example:
User: "Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait in the entrance to the other room."
You: {{"destination": "entrance to the other room under the balcony", "sub-instructions": ["Walk into the living room", "keep walking straight past the living room", "walk into the entrance under the balcony", "wait in the entrance to the other room"], "state-constraints": {{"0": [["location constraint", "living room"]], "1": [["location constraint", "living room"]], "2": [["location constraint", "balcony"], ["object constraint", "entrance"]], "3": [["location constraint", "other room"], ["object constraint", "entrance"]]}}, "decisions": {{"0": {{"landmarks": [["living room", "approach"]], "directions": ["forward"]}}, "1": {{"landmarks": [["living room", "move away"]], "directions": ["forward"]}}, "2": {{"landmarks": [["balcony", "approach"], ["entrance", "approach"]], "directions": ["forward"]}}, "3": {{"landmarks": [["other room", "approach"], ["entrance", "approach"]], "directions": ["forward"]}}}}}}
ATTENTION:
1. constraint type: location constraint is for room type, object constraint is for object type, directions constraint. Don't confuse object constriant with location constraint!
2. landmark choice: approach, move away, approach then move away
3. direction choice: left, right, forward
4. The landmark and constraint object should not accurately describe the orientation and order. Here arre examples about landmark: "second step from the top" -> "step"(neglect order and position relation); "room directly ahead" -> "room"; "right bedroom door" -> "bedroom door"
"""


def get_reply(client, id, prompt, max_retry_times=5, retry_interval_initial=1):
    global dones
    retry_interval = retry_interval_initial
    reply = None
    for _ in range(max_retry_times):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # model="gpt-3.5-turbo",
                model="gpt-4",
                temperature=0
            )
            reply = eval(chat_completion.choices[0].message.content.strip())
            res = {str(id): reply}
            with open(TEMP_SAVE_PATH, 'a') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            
            dones += 1
            print(id, dones)
            
            return id, reply
        except:
            print(f"{id} Error, retrying...")
            time.sleep(max(retry_interval, 10))
            retry_interval *= 2

    dones += 1
    print(id, dones)
    return id, reply


def check_exist_replys(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing_data = json.load(f)
        exist_keys = list(existing_data.keys())
        exist_keys = [int(k) for k in exist_keys]
        
        return exist_keys
    else:
        return []


def generate_prompts(exist_replys, num=None):
    with gzip.open(R2R_VALUNSEEN_PATH, 'r') as f:
        eps_data = json.loads(f.read().decode('utf-8'))
    eps_data = eps_data["episodes"]
    eps_data = [item for item in eps_data if item["episode_id"] not in exist_replys]
    if len(eps_data) == 0:
        print("all episodes are generated")
        return {}
    random.shuffle(eps_data)
    episodes = random.sample(eps_data, min(num, len(eps_data)))
    prompts = {}
    for episode in episodes:
        id = episode["episode_id"]
        instruction = episode["instruction"]["instruction_text"]
        prompts[id] = prompt_template + f"\"\"\"{instruction}\"\"\""
    
    return prompts


def generate_specific_prompts(id: int):
    with gzip.open(R2R_VALUNSEEN_PATH, 'r') as f:
        eps_data = json.loads(f.read().decode('utf-8'))
    eps_data = eps_data["episodes"]
    prompts = {}
    for episode in eps_data:
        if episode["episode_id"] == id:
            instruction = episode["instruction"]["instruction_text"]
            prompts[id] = prompt_template + f"\"\"\"{instruction}\"\"\""
    
    return prompts


def regenerate_exist_keys(exist_replys):
    with gzip.open(R2R_VALUNSEEN_PATH, 'r') as f:
        eps_data = json.loads(f.read().decode('utf-8'))
    eps_data = eps_data["episodes"]
    eps_data = [item for item in eps_data if item["episode_id"] in exist_replys]
    prompts = {}
    for episode in eps_data:
        id = episode["episode_id"]
        instruction = episode["instruction"]["instruction_text"]
        prompts[id] = prompt_template + f"\"\"\"{instruction}\"\"\""
    
    return prompts


def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    
    return sub_strings


def main():
    client = OpenAI(
        api_key="dc4e8ca91bb9509727662d60ff4ad16b",
        base_url="https://flag.smarttrot.com/v1/"
    )
    
    if os.path.exists(DIR_NAME):
        all_exist_files = sorted(os.listdir(DIR_NAME), key=natural_sort_key, reverse=True)
        if len(all_exist_files) > 0:
            current_file = all_exist_files[0]
            file_path = os.path.join(DIR_NAME, current_file)
        else:
            file_path = ''
    else:
        os.makedirs(DIR_NAME, exist_ok=True)
        file_path = ''
    
    exist_replys = check_exist_replys(path=file_path)
    prompts = generate_prompts(exist_replys, num=1)
    if len(prompts) == 0:
        return
    # prompts = generate_specific_prompts(id=164)
    # prompts = regenerate_exist_keys(exist_replys)
    
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = [executor.submit(get_reply, client, id, prompt) for id, prompt in prompts.items()]
        query2res = {job.result()[0]: job.result()[1] for job in as_completed(results)}
    
    # sorted_data = {k: query2res[k] for k in sorted(query2res, key=int)}
    # length = len(sorted_data)
    # new_filename = FILE_NAME + str(length) + "_version2.json"
    # with open(os.path.join(DIR_NAME, new_filename), 'w') as f:
    #     json.dump(sorted_data, f, indent=4, ensure_ascii=False)
        
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        for key, value in query2res.items():
            if str(key) not in existing_data:
                existing_data[str(key)] = value
        sorted_data = {k: existing_data[k] for k in sorted(existing_data, key=int)}
        
        # avoid overwrite
        length = len(sorted_data)
        new_filename = FILE_NAME + str(length) + ".json"
        with open(os.path.join(DIR_NAME, new_filename), 'w') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    else:
        sorted_data = {k: query2res[k] for k in sorted(query2res, key=int)}
        length = len(sorted_data)
        new_filename = FILE_NAME + str(length) + ".json"
        with open(os.path.join(DIR_NAME, new_filename), 'w') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()