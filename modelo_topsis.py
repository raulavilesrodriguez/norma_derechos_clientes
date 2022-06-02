import skcriteria as skc
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.madm import similarity  # here lives TOPSIS
from skcriteria.pipeline import mkpipe  # this function is for create pipelines
import matplotlib.pyplot as plt

def main():
    # Matriz de alternativas
    matrix = [
        [383052, 17760, 523274, 150000],  # alternativa 1: ModificaciónNormativa
        [116093, 24480, 790233, 0.1],  # alternativa 2: Difusión y ActividadesControl
        [0, 14400, 906326, 0.1], # alternativa 3: No Acción
    ]

    objectives = [max, min, min, min]

    dm = skc.mkdm(
        matrix,
        objectives,
        weights=[0.25, 0.25, 0.25, 0.25],
        alternatives=[
            'ModNormativa', 
            'Comunicación',  
            'NoAcción'
        ],
        criteria=[
            'BenefCliente',
            'CostoArcotel',
            'CostoCliente',
            'CostoPrestador'
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