import skcriteria as skc
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.madm import similarity  # here lives TOPSIS
from skcriteria.pipeline import mkpipe  # this function is for create pipelines
import matplotlib.pyplot as plt

def main():
    # Matriz de alternativas
    matrix = [
        [11147, 3, 1480, 4, 2],  # alternativa 1: ModificaciónNormativa
        [14332, 2, 2400, 2, 1],  # alternativa 2
        [14332, 4, 840, 2, 1], # alternativa 3
        [15924, 1, 1200, 1, 1], # alternativa 4
    ]

    objectives = [min, max, min, max, min]

    dm = skc.mkdm(
        matrix,
        objectives,
        weights=[0.30, 0.20, 0.10, 0.30, 0.10],
        alternatives=[
            'ModificaciónNormativa', 
            'ActividadesControl', 
            'Difusión', 
            'NoAcción'
        ],
        criteria=[
            'NumReclamos',
            'ImagenInstitucional',
            'CostoArcotel',
            'Transparencia',
            'CostoPrestadores'
        ],
    )

    pipe = mkpipe(
    invert_objectives.MinimizeToMaximize(),
    scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
    scalers.SumScaler(target="weights"),  # and this transform the weights
    similarity.TOPSIS(),
    )

    rank = pipe.evaluate(dm)
    print(rank)
    print(rank.e_)
    print("Ideal:", rank.e_.ideal)
    print("Anti-Ideal:", rank.e_.anti_ideal)
    print("Similarity index:", rank.e_.similarity)

if __name__ == "__main__":
    main()