from abc import ABC, abstractmethod
import torch 

class Evaluator(ABC):

    def __init__(self, metrics):
         
        self.metrics = {}

        if isinstance(metrics, str):
            metrics = [metrics]

        if not isinstance(metrics, list):
            raise ValueError("metrics debe ser str o list de strings")

        for metric in metrics:
            if not isinstance(metric, str):
                raise TypeError("Cada métrica debe ser un string")

            self.metrics[metric] = self.mapping_metrics(metric)


    def mapping_metrics(self, name):
        if name not in self.all_metrics:
            raise ValueError(f"Métrica desconocida: {name}")

        return self.all_metrics[name]


    def __call__(self, trainer):

        trainer.eval()

        with torch.no_grad():
            for function in self.metrics.values():
                
                function(trainer)
