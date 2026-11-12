
from importlib import import_module
from collections.abc import Callable
import torch 


def resolve_callables_in_mapping(obj):
    """
    Recorre recursivamente una estructura de configuración y resuelve
    posibles referencias a objetos Python.

    Soporta diccionarios y listas anidadas. Los valores simples se pasan
    a `resolve_callable()`.

    Ejemplo:
        {
            "optimizer": {
                "class": "torch.optim:Adam"
            }
        }

    se transforma en:

        {
            "optimizer": {
                "class": torch.optim.Adam
            }
        }
    """

    if isinstance(obj, dict):
        return {
            key: resolve_callables_in_mapping(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            resolve_callables_in_mapping(value)
            for value in obj
        ]

    return resolve_callable(obj)


def resolve_callable(obj):
    """
    Convierte una referencia textual `"modulo:atributo"` en el objeto
    Python correspondiente.

    - `None` se devuelve sin cambios.
    - Los objetos que ya son callable se devuelven sin cambios.
    - Strings sin `:` se devuelven sin cambios.

    Ejemplo:
        "torch.optim:Adam" -> torch.optim.Adam
    """

    if obj is None or isinstance(obj, Callable):
        return obj

    if isinstance(obj, str) and ":" in obj:
        module_name, attribute_name = obj.split(":", 1)
        module = import_module(module_name)
        return getattr(module, attribute_name)

    return obj


def get_available_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')