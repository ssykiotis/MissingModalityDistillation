from abc import ABC, abstractmethod

class GeneralDatasetParser(ABC):

    @abstractmethod
    def parse_data(self,shuffle:bool = False):
        pass


    @abstractmethod
    def train_test_split(self):
        pass

    @abstractmethod
    def get_dataset(self, dataset_type:str, training_mode:str):
        pass
    
    @abstractmethod
    def read_data(self, dataset_type:str):
        pass