from module import (
    Med_HOModule,
    Med_NMModule,
    Med_KGRMModule,
    Med_DVMModule,
    Med_LRMModule,
)
from config import Config
from utils.singleton import Singleton


@Singleton
class ModuleManager:
    def __init__(self):
        pass

    def setup(self, config: Config = None):
        self.Med_HOModule = Med_HOModule(config)
        self.Med_NMModule = Med_NMModule(config)
        self.Med_KGRMModule = Med_KGRMModule(config)
        self.Med_DVMModule = Med_DVMModule(config)
        self.Med_LRMModule = Med_LRMModule(config)

    def mapper(self):
        return (
            self.Med_HOModule,
            self.Med_NMModule,
            self.Med_KGRMModule,
            self.Med_DVMModule,
            self.Med_LRMModule,
        )
