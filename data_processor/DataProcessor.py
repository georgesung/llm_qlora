from abc import ABC, abstractmethod


class DataProcessor(ABC):
    @abstractmethod
    def get_data(self):
        pass
