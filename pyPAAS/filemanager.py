"""
A class to manage folders and files 
"""
from pathlib import Path

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.

    More info at https://refactoring.guru/es/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


        
class FileManager(metaclass=SingletonMeta):

    def __init__(self):

        self.root_name = "data/"
        self.root_path = Path(f'{self.root_name}')
        self.results_folder = self.root_path/f'plots/'
        self.random_model  = self.root_path/f'random.pkl'
        self.ranking_paas = self.root_path/f'prioritized_PAAS.pkl'
        self.ranking_mie = self.root_path/f'prioritized_MIE.pkl'
        self.data_file = self.root_path/f'data.pkl'

        self.results_folder.mkdir(parents=True, exist_ok=True)


