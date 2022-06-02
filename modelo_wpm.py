import skcriteria as skc
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.madm import simple  # Método de producto ponderado (WPM)  
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

    # Normalización de los datos de entrada
    inverter = invert_objectives.MinimizeToMaximize()
    dmt = inverter.transform(dm)

    scaler = scalers.SumScaler(target="both")
    dmt = scaler.transform(dmt)

    # Uso del Método de producto ponderado (WPM)
    dec = simple.WeightedSumModel()
    rank = dec.evaluate(dmt)  # Se evalua los datos normalizados
    print(rank)
    print(rank.e_.score)

    # graficar alternativas y criterios
    dm.plot()

    # Se crea 2 ejes para la visualización
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # En el primer eje se grafica el criterio KDE
    dmt.plot.kde(ax=axs[0])
    axs[0].set_title("Criterios")

    # En el segundo eje se grafica los diferentes pesos
    dmt.plot.wbar(ax=axs[1])
    axs[1].set_title("Pesos")

    # Se ajusta el layout al contenido
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()