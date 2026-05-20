from config import Config
from utils.prompt_build import build_RAG_prompt
from utils.singleton import Singleton


@Singleton
class Med_LRMModule:
    def __init__(self, config: Config = None):
        self.config = config

    def process(self, gold_triplets, entity_description, origin_data):
        return build_RAG_prompt(gold_triplets, entity_description, origin_data, self.config)
