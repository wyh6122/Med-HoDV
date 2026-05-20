import json
import requests

from .prompt_template import (
    Med_HO_IN_template,
    Med_DVM_PLAN_template,
    Med_Reader_prompt_template,
)
from config import Config


def build_Med_HO_prompt(data):
    return Med_HO_IN_template.format(data)


def build_Med_DVM_plan_prompt(query, hypothesis):
    return Med_DVM_PLAN_template.format(query, hypothesis)


def build_RAG_prompt(gold_triplets, entity_description, origin_data, config: Config = None):
    RAG_prompt = med_reader_prompt_build(
        gold_triplets,
        entity_description,
        origin_data,
        config.configKG.knowledge_format_choice
    )
    print("返回Prompt", RAG_prompt)
    return RAG_prompt


def build_RAGent_prompt(gold_triplets, entity_description, origin_data, config: Config = None):
    return _format_med_hodv_knowledge(
        gold_triplets,
        entity_description,
        config.configKG.knowledge_format_choice
    )


def med_reader_prompt_build(gold_triplets, entity_description, query, knowledge_format_choice):
    knowledge = _format_med_hodv_knowledge(gold_triplets, entity_description, knowledge_format_choice)
    return Med_Reader_prompt_template.format(knowledge, query)


def _format_med_hodv_knowledge(gold_triplets, entity_description, knowledge_format_choice):
    knowledge_parts = []

    if gold_triplets:
        if knowledge_format_choice == 'triplets':
            for fact in gold_triplets:
                knowledge_parts.append("{}和{}的关系是{}".format(fact['S'], fact['O'], fact['P']))
        elif knowledge_format_choice == 'paths':
            for fact in gold_triplets:
                modified_string = ""
                for item in fact:
                    tlist = str(item).split(":")
                    if tlist[0] == "i":
                        modified_string += "->" + tlist[1] + "->"
                    elif tlist[0] == "o":
                        modified_string += "<-" + tlist[1] + "<-"
                    else:
                        modified_string += tlist[0]
                knowledge_parts.append(modified_string)

    if entity_description:
        for key, value in entity_description.items():
            knowledge_parts.append("{}的定义是{};".format(key, value))

    return "\n".join(knowledge_parts)


def talk_to_LLM(context="您好，请进行医学分析。", temperature=0.5, response_num=3, max_tokens=100):
    url = 'http://localhost:8080/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "string",
        "messages": [
            {
                "role": "user",
                "content": context
            }
        ],
        "do_sample": True,
        "temperature": temperature,
        "n": response_num,
        "max_tokens": max_tokens,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_data = response.json()
    message = []
    if 'detail' in response_data:
        print("Encounted Validation Error---->msg:{}  type:{}".format(response_data['detail'][0]['msg'],
                                                                      response_data['detail'][0]['type']))
    else:
        for msg in response_data['choices']:
            message.append(msg['message']['content'])
    return message


def LLM_process_NER(query):
    query = str(query).replace("\n", "")
    instruction = (
        '从给定文本中抽取可能的医学相关实体，只抽取药物、疾病、症状、检查、治疗方案等实体。'
        '请严格输出 JSON：{"output":["实体1","实体2"]}，不要输出额外文字。'
    )
    prompt = instruction + "\n输入：" + str(query) + "\n输出："
    response = talk_to_LLM(context=prompt, temperature=0.6)

    ner = []
    for item in response:
        item = str(item).replace("\n", "")
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            continue
        if 'output' in item:
            for entity in item['output']:
                if entity not in ner:
                    ner.append(entity)

    res = ''
    for entity in ner:
        res = res + entity + ","
    print("LLM抽取结果:")
    print(res)
    return res
