from config import Config
from utils.cut_stop_words import *
from sentence_transformers import SentenceTransformer
import json
import os
import re
import numpy as np
from openai import OpenAI
from dashscope import Generation
from utils.singleton import Singleton
from utils.prompt_build import build_Med_DVM_plan_prompt, talk_to_LLM

@Singleton
class Med_DVMModule:
    def __init__(self,config: Config = None):
        self.FILTER_ENABLE = config.configFilter.FILTER_ENABLE
        self.knowledge_format_choice = config.configKG.knowledge_format_choice
        self.TOP_SIM = config.configFilter.TOP_SIM
        self.med_dvm_alpha = config.configFilter.med_dvm_alpha
        self.med_dvm_beta = config.configFilter.med_dvm_beta
        self.med_dvm_top_k = config.configFilter.med_dvm_top_k
        self.med_dvm_plan_max_token_num = config.configFilter.med_dvm_plan_max_token_num
        self.HyDE = config.configHO.HyDE
        self.USE_CHUNK = config.configFilter.USE_CHUNK
        self.ENCODING_CHOICE = config.configKG.ENCODING_CHOICE
        self.LLM_type = config.configHO.LLM_type

        self.stop_words_cache = load_stop_words()
        # ENCODING_Choice_list = ["CME", "Piccolo", "GTE"]
        if self.ENCODING_CHOICE == "Piccolo":
            self.encoding_model = SentenceTransformer('./models/embedding_model/piccolo_checkpoint/')
        if self.ENCODING_CHOICE == "GTE":
            self.encoding_model = SentenceTransformer('./models/embedding_model/gte_checkpoint/')
        if self.LLM_type == 'GPT3.5' or self.LLM_type == 'GPT4':
            self.openai_client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                base_url="https://api.chatanywhere.com.cn/v1"
            )
        elif self.LLM_type == 'Ali':
            assert os.environ['DASHSCOPE_API_KEY'] is not None
            self.openai_client = Generation()
        else:
            self.openai_client = None

    
    def process(self,gold_triplets,origin_data,hyde_data):
        if gold_triplets == [] or not self.FILTER_ENABLE:
            return gold_triplets
        return self.process_med_dvm(gold_triplets, origin_data, hyde_data)

    def process_med_dvm(self, gold_triplets, origin_data, hyde_data):
        medical_hypothesis = self._extract_medical_hypothesis(origin_data, hyde_data)
        clinical_sub_questions, relation_keywords = self._generate_reasoning_plan(origin_data, medical_hypothesis)
        if not clinical_sub_questions:
            clinical_sub_questions = [origin_data]

        path_text_list = [self._path_to_text(path) for path in gold_triplets]
        path_embedding = self.encoding_model.encode(path_text_list, normalize_embeddings=True)
        question_embedding = self.encoding_model.encode(clinical_sub_questions, normalize_embeddings=True)
        semantic_scores = np.matmul(path_embedding, question_embedding.T).mean(axis=1)

        scored_triplets = []
        for idx, path in enumerate(gold_triplets):
            structure_score = self._structure_score(path, relation_keywords)
            score = self.med_dvm_alpha * float(semantic_scores[idx]) + self.med_dvm_beta * structure_score
            scored_triplets.append((score, path))

        scored_triplets.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in scored_triplets[:min(self.med_dvm_top_k, len(scored_triplets))]]

    @staticmethod
    def _extract_medical_hypothesis(origin_data, hyde_data):
        if hyde_data.startswith(origin_data):
            return hyde_data[len(origin_data):].strip()
        return hyde_data

    def _generate_reasoning_plan(self, origin_data, hyde_data):
        plan_prompt = build_Med_DVM_plan_prompt(origin_data, hyde_data)
        plan_output = ""
        try:
            if self.LLM_type == 'GPT4':
                model = "gpt-4-vision-preview"
            elif self.LLM_type == 'GPT3.5':
                model = "gpt-3.5-turbo-1106"
            else:
                model = None

            if model is not None:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": plan_prompt},
                            ],
                        }
                    ],
                    max_tokens=self.med_dvm_plan_max_token_num,
                    stream=False
                )
                plan_output = response.choices[0].message.content
            elif self.LLM_type == 'Ali':
                response = self.openai_client.call(
                    Generation.Models.qwen_max,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": plan_prompt},
                    ],
                    result_format='message',
                    max_tokens=self.med_dvm_plan_max_token_num,
                )
                plan_output = response.output.choices[0].message.content.strip()
            else:
                plan_output = talk_to_LLM(plan_prompt, response_num=1, max_tokens=self.med_dvm_plan_max_token_num)[0]
        except Exception as e:
            print(f"Med-DVM planning error: {e}")

        return self._parse_reasoning_plan(plan_output)

    @staticmethod
    def _parse_reasoning_plan(plan_output):
        if not plan_output:
            return [], []

        try:
            json_text = re.search(r"\{.*\}", plan_output, re.S).group(0)
            plan = json.loads(json_text)
            clinical_sub_questions = plan.get("clinical_sub_questions", [])
            relation_keywords = plan.get("relation_keywords", [])
            return clinical_sub_questions, relation_keywords
        except Exception:
            clinical_sub_questions = []
            relation_keywords = []
            for line in plan_output.splitlines():
                line = line.strip("- \t\r\n")
                if not line:
                    continue
                if "?" in line or "？" in line:
                    clinical_sub_questions.append(line)
                if "关系" in line or "关键词" in line:
                    relation_keywords.extend(re.split(r"[,，、;；\s]+", line))
            relation_keywords = [
                item for item in relation_keywords
                if item and item not in {"关系", "关键词", "关系关键词"}
            ]
            return clinical_sub_questions, relation_keywords

    @staticmethod
    def _path_to_text(path):
        if isinstance(path, dict):
            return path['S'] + path['P'] + path['O']

        path_text = ""
        for item in path:
            tlist = str(item).split(":")
            if tlist[0] == "i":
                path_text += "->" + tlist[1] + "->"
            elif tlist[0] == "o":
                path_text += "<-" + tlist[1] + "<-"
            else:
                path_text += tlist[0]
        return path_text

    @staticmethod
    def _path_relations(path):
        if isinstance(path, dict):
            return [path['P']]

        relations = []
        for item in path:
            tlist = str(item).split(":")
            if len(tlist) > 1 and tlist[0] in {"i", "o"}:
                relations.append(tlist[1])
        return relations

    def _structure_score(self, path, relation_keywords):
        relation_keywords = {str(item).strip() for item in relation_keywords if str(item).strip()}
        if not relation_keywords:
            return 0.0

        relations = self._path_relations(path)
        matched_keywords = set()
        for keyword in relation_keywords:
            for relation in relations:
                relation = str(relation).strip()
                if keyword == relation or keyword in relation or relation in keyword:
                    matched_keywords.add(keyword)
                    break

        return len(matched_keywords) / len(relation_keywords)
