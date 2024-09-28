import os
import inspect
import sys

def get_current_folder():
    # Obtener el stack frame actual
    frame = inspect.currentframe()
    # Obtener el stack frame del llamador
    caller_frame = inspect.getouterframes(frame, 2)
    # Obtener la ruta del archivo del llamador
    ruta_script = caller_frame[1].filename
    # Obtener el directorio padre
    directorio_actual = os.path.dirname(os.path.abspath(ruta_script))
    # Devolver la ruta del directorio padre
    return directorio_actual


def get_name_parent_script():
    script_path = sys.argv[0]
    script_name = os.path.basename(script_path)
    if script_name.endswith('.py'):
        return script_name[:-3]  # Elimina la extensión .py
    else:
        return script_name
EXPERIMENTS_FOLDER = 'experiments'
RESULTS_FOLDER = 'results'

def get_experiment_folder(dataset_2d, dataset_3d, bias:int=1, root:str = None):
    
    if root is None:
        root = f'{get_current_folder()/EXPERIMENTS_FOLDER/RESULTS_FOLDER}'
    
    if isinstance(dataset_2d, str) and isinstance(dataset_3d, str):
        folder = f'{root}/{dataset_2d}_vs_{dataset_3d}/{bias}'
    else:
        folder = f'{root}/{get_name_parent_script()}/{bias}'

    # if os.path.exists(folder):
    #     folder = get_experiment_folder(dataset_2d, dataset_3d, bias+1, root=root)
    # os.makedirs(folder, exist_ok=True)
    return folder




