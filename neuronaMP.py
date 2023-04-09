#Libreria que nos ayuda a obtener la informacion sobre los datos
#De cancer de mama
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from neuronaMP_class import  MPNeuron
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd


if __name__ == '__main__':
    breast_cancer=load_breast_cancer()
    x=breast_cancer.data#Obtenemos los datos
    Y=breast_cancer.target#Obtenemos la etiqueta y nos da que tiempo de enfermedad es
    # visualizamos el conjuntos de datos 
    # Transformamos la informacion en un pandas para verlo mejor
    df=pd.DataFrame(x, columns=breast_cancer.feature_names)
    # print(df)
    
    # dividos en subconjuntos de datos 
    x_train,x_test,y_train,y_test=train_test_split(df,Y,stratify=Y)
    print("Tamaño de conjuntos de datos de entrenamiento :",len(x_train))
    print("Tamaño de conjuntos de datos de prueba:",len(x_test))    
    # Transformamos las caracteríticas de entrada a un valor binario
    #pd.cut que se utiliza para segmentar y clasificar datos en contenedores o intervalos.
    x_train_bin = x_train.apply(pd.cut, bins=2, labels=[1, 0])
    x_test_bin = x_test.apply(pd.cut, bins=2, labels=[1, 0])
    print(x_train_bin)
    # Instanciamos el modelo MPNeuron
    mp_neuron = MPNeuron()

    # Encontramos el threshold óptimo en la clase nos da la columna donde tenga los mejores resultados
    mp_neuron.fit(x_train_bin.to_numpy(), y_train)
    print( mp_neuron.threshold)
    # Realizamos predicciones para ejemplos nuevos que no se encuentran en el conjunto de datos de entrenamiento
    Y_pred = mp_neuron.predict(x_test_bin.to_numpy())
    print(Y_pred)
    # Calculamos la exactitud de nuestra predicción
    print(accuracy_score(y_test, Y_pred))
    # Calculamos la matriz de confusión
    
    
    print(confusion_matrix(y_test, Y_pred))
    
    