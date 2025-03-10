import requests
import sys
import json
import re
from openai import OpenAI

# model_path = sys.argv[1]

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""
URL = "http://localhost:7777/v1/chat/completions"
MSG={"model": "tgi", 
     "messages": [ 
                    { "role": "system", "content": system_prompt }, 
                    { "role": "user", "content": "%s" } 
     ], 
     "stream": False, "max_tokens": 4096, "temperature": 0.01 }
MSG = json.dumps(MSG)

def extract_final_answer2(question):
    msg = "only output result without other words. what's the final answer according to following:\n"
    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
                api_key="YUSujgDrR4WMhdgoi2LcTG1ay68hPLEG",
                    base_url="https://api.deepinfra.com/v1/openai",
                    )

    chat_completion = openai.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[{"role": "user", "content": msg + question}],
                    )

    result = chat_completion.choices[0].message.content
    # print(result)
    # print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)    
    return result

def extract_final_answer(completion):
    # search answer tag
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL,
    )

    answer = answer_match.group(1) if answer_match else None
    return answer

def _match_answer(completion, oracle_answer):
    # search answer tag
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL,
    )

    answer = answer_match.group(1) if answer_match else None
    reward = 0
    if answer is not None:
        if answer == oracle_answer:
            reward = 1.0
        elif oracle_answer in answer:
            reward = 0.5
        else:
            reward = 0.01
    return reward

def _match_final(result1, result2, id):
    try:
        ret = eval(str(result1).replace(",", "").replace(" ", "")) == eval(str(result2).replace(",", "").replace(" ", ""))
    except Exception as e:
        print(f"{result1} and {result2} ==> {e}")
        return -1
    return 1 if ret else 0

def eval_file(test_data_file:str, outfile:str = 'eval_results.json'):
    results = []
    with open(test_data_file, 'r') as rfl:
        for _idx, _line in enumerate(rfl.readlines()):
            _data = json.loads(_line)
            _question = _data['question']
            _answer = _data['answer']
            _msg = MSG % _question
            _resp = requests.post(URL, headers={'Content-Type': 'application/json'}, data=_msg)
            _result = {}
            _result.update(_data)
            _result['generated'] = ''
            try:
                _resp_json = json.loads(_resp.text)
                _content = _resp_json['choices'][0]['message']['content']
                _result['generated'] = _content
                _result['final_answer'] = extract_final_answer(_content)
                _result['autoreview'] = _match_answer(_content, _answer)
                _result['autofinal'] = _match_final(_result['final_answer'], _answer, _data['id'])
            except Exception as e:
                print("id: ", _data['id'], "==>", e)
            results.append(_result)

            if (_idx + 1) % 10 == 0:
                with open(outfile, 'w') as wfl:
                    json.dump(results, wfl)
                print("processed records: ", _idx + 1)
    
    with open(outfile, 'w') as wfl:
        json.dump(results, wfl)
    return results

if __name__ == '__main__':
    import os 
    outfile = "eval_results.json"
    _tag = os.environ.get('TAG')
    if _tag:
        outfile = 'eval_results_%s.json' % _tag
    test_data_file = "data.json" if len(sys.argv) < 2 else sys.argv[1]
    eval_file(test_data_file, outfile=outfile) 