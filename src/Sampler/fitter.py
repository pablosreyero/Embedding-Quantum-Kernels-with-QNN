import numpy as np
from icecream import ic
from src.dataReuploading import Constant

def fit_argument(func, target=0.5, tolerance=0.05):

    
    radius = np.arange(0, 2, tolerance)

    def indice_mas_cercano_a_target(array):
        indice = None
        distancia_minima = float('inf')

        for i in range(len(array)):
            distancia_actual = abs(array[i] - target)
            if distancia_actual < distancia_minima:
                distancia_minima = distancia_actual
                indice = i

        return indice

    perc = []
    for x in radius:
        a, labels = func(x)
        perc.append((labels == 1).sum() / len(labels))

    ix = indice_mas_cercano_a_target(perc)
    closer_radius = radius[ix]
    
    # Check consistency
    
    points, labels = func(closer_radius)
    
    constant = Constant()
    constant.train(points, labels)
    acc = constant.get_accuracy(points, labels)
    assert target - tolerance <= acc <= target + tolerance
    
    
    return closer_radius