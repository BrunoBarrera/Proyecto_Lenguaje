from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import pickle
import numpy as np
import statistics
import sklearn.metrics as met
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test
class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set
  
# def Entrenamiento(X_train, y_train):
#     i=0
#     Class1=[]
#     Class2=[]
#     Class3=[]
#     Class4=[]
#     Class5=[]
#     Diferencias_Totales=[]
#     # for opinion in val_set.X_train:
#     for opinion in X_train:
#         PalabraDesconocida=0
#         titulo_Positivo=0
#         titulo_Negativo=0
#         opinion_positivo=0
#         opinion_negativo=0
#         Diferencia_total_positivo=0
#         Diferencia_total_negativo=0
#         #Para los Titulos
#         #print(opinion[0])
#         for word in opinion[0]:
#             try:
#                 Sentimiento=dic.get(word)[0]
#                 # if(Sentimiento=="Alegría" or Sentimiento =="Sorpresa"): # Alegría y Sorpresa
#                 titulo_Positivo+=dic.get(word)[0]
#                 # else:                                                   # Enojo Miedo Repulsión Tristeza
#                 titulo_Negativo+=dic.get(word)[1]
#             except:
#                 PalabraDesconocida+=1
#                 # print("No exite esa palabra en el diccionario")
#         #Para las Opiniones
#         for word in opinion[1]:
#             try:
#                 Sentimiento=dic.get(word)[1]
#                 # if(Sentimiento=="Alegría" or Sentimiento =="Sorpresa"): # Alegría y Sorpresa
#                 opinion_positivo+=dic.get(word)[0]

#                 # else:                                                   # Enojo Miedo Repulsión Tristeza
#                 opinion_negativo+=dic.get(word)[1]
#             except:
#                 PalabraDesconocida=PalabraDesconocida
#         Diferencia_total_positivo=titulo_Positivo+opinion_positivo
#         Diferencia_total_negativo=titulo_Negativo+opinion_negativo
#         DiferenciaTotal=opinion_positivo-opinion_negativo
#         Diferencias_Totales.append(DiferenciaTotal)
#     #     if y_train[i]==1:
#     #         Class1.append(DiferenciaTotal)
#     #     elif y_train[i]==2:
#     #         Class2.append(DiferenciaTotal)
#     #     elif y_train[i]==3:
#     #         Class3.append(DiferenciaTotal)
#     #     elif y_train[i]==4:
#     #         Class4.append(DiferenciaTotal)
#     #     elif y_train[i]==5:
#     #         Class5.append(DiferenciaTotal)
#     #     i+=1
#     # return Class1, Class2, Class3, Class4, Class5
#     return Diferencias_Totales

def Entrenamiento(X_train, y_train):
    i=0
    DiferenciaTotal=[]
    X_Diferencia_titulo=[]
    X_Diferencia_opinion=[]
    DifPosOp=[]
    DifNegOp=[]
    Data=[]
    # for opinion in val_set.X_train:
    for opinion in X_train:
        PalabraDesconocida=0
        titulo_Positivo=0
        titulo_Negativo=0
        opinion_positivo=0
        opinion_negativo=0
        #Para los Titulos
        # print(opinion)
        for word in opinion[0]:
            # print(word)
            try:
                Sentimiento=dic.get(word)[0]
                # if(Sentimiento=="Alegría" or Sentimiento =="Sorpresa"): # Alegría y Sorpresa
                titulo_Positivo+=dic.get(word)[0]
                # else:                                                   # Enojo Miedo Repulsión Tristeza
                titulo_Negativo+=dic.get(word)[1]
            except:
                PalabraDesconocida+=1
                # print("No exite esa palabra en el diccionario")
        #Para las Opiniones
        for word in opinion[1]:
            try:
                Sentimiento=dic.get(word)[1]
                # if(Sentimiento=="Alegría" or Sentimiento =="Sorpresa"): # Alegría y Sorpresa
                opinion_positivo+=dic.get(word)[0]

                # else:                                                   # Enojo Miedo Repulsión Tristeza
                opinion_negativo+=dic.get(word)[1]
            except:
                PalabraDesconocida=PalabraDesconocida
                # PalabraDesconocida+=1
                # print("No exite esa palabra en el diccionario")
        # print("Pos:"+str(opinion_positivo))
        # print("Neg:"+str(opinion_negativo))
        # val_set.y_train
        # Data.append([opinion_positivo,opinion_negativo,y_train])
        Data.append([opinion_positivo,opinion_negativo])
        
        i+=1
        # print("Positivo acumulado:"+str(titulo_Positivo)+", Negativo acumulado:"+str(titulo_Negativo))
        # print("Tam titulo:"+str(len(opinion[0]))+",  Descpmocido "+str(PalabraDesconocida))
        X_Diferencia_titulo.append(titulo_Positivo-titulo_Negativo)
        DifPosOp.append(opinion_positivo-opinion_negativo)
        DifNegOp.append(opinion_negativo-opinion_positivo)
        X_Diferencia_opinion.append(opinion_positivo-opinion_negativo)
        DiferenciaTotal.append((titulo_Positivo-titulo_Negativo)*0.70+(opinion_positivo-opinion_negativo)*0.30)
        # print("Diferencia Titulo["+str(i)+"]="+str(X_Diferencia_titulo[len(X_Diferencia_titulo) - 1]))
        # print("Diferencia Opinion["+str(i)+"]="+str(X_Diferencia_opinion[len(X_Diferencia_opinion) - 1]))
        # print("Palabras Desconocidas: "+str(PalabraDesconocida))
    return Data,DiferenciaTotal,X_Diferencia_opinion


def Evaluador(Diferencia):
    i=0
    ValoracionOpinion=[]
    for df in Diferencia:
        if df <= -0.9:
            ValoracionOpinion.append(1)
        elif  df <= -0.4:
            ValoracionOpinion.append(2)
        elif  df <= -0.2:
            ValoracionOpinion.append(3)
        elif  df <= -0.1:
            ValoracionOpinion.append(4)
        elif df>-0.1:
            ValoracionOpinion.append(5)
        else:
            ValoracionOpinion.append(2)
        # print("Valoracion Predicha Opinion["+str(i)+"]: "+str(ValoracionOpinion[len(ValoracionOpinion) - 1])+", valor real: "+str(y_train[i]))
        i+=1
    return ValoracionOpinion
  
with open("matriz_opiniones_titulos_lematizados.pickle", "rb") as f:
    obj = pickle.load(f)
df=pd.DataFrame(obj)

with open("Dictionary.pickle", "rb") as f:
    dic = pickle.load(f)

#Guardamos los datos de entrenamiento en X
X=[]
for i in range(len(obj)):
    if i!=0:
        x1=str(obj[i][0])
        x2=str(obj[i][1])
        X.append([x1.split(),x2.split()])

#Guardamos las etiquetas en y
y=[]
for i in range(len(obj)):
    if i!=0:
        y.append(obj[i][2])

#Dividimos en train y test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2,random_state=0,shuffle= True)

validation_sets = []
kf = KFold(n_splits=5)
c=0
for train_index, test_index in kf.split(X_train):
    c=c+1
    # X_train_v, X_test_v = X_train[train_index], X_train[test_index]
    X_train_v = np.array(X_train, dtype=object)[train_index.astype(int)]
    X_test_v = np.array(X_train, dtype=object)[test_index.astype(int)]
    # y_train_v, y_test_v = y_train[train_index], y_train[test_index]
    y_train_v = np.array(y_train, dtype=object)[train_index.astype(int)]
    y_test_v = np.array(y_train, dtype=object)[test_index.astype(int)]
    validation_sets.append(validation_set(X_train_v, y_train_v, X_test_v, y_test_v))
my_test_set = test_set(X_test, y_test)	
my_data_set = data_set(validation_sets, my_test_set) 

Umbrales1=[]
Umbrales2=[]
Umbrales3=[]
Umbrales4=[]
Umbrales5=[]
UmbralesFinales=[]
pliegue=1
for val_set in my_data_set.validation_set:
    print("Pliegue: ",pliegue)
    Umbral1=[]
    Umbral2=[]
    Umbral3=[]
    Umbral4=[]
    Umbral5=[]
    #Datos de entrenamiento
    Data,DiferenciaTotal,X_Diferencia_opinion= Entrenamiento(val_set.X_train,val_set.y_train)
    #umb1=[min(Umbral1), max(Umbral1)]
    #umb1=[statistics.mode(Umbral1)]
    #Umbrales1.append(umb1)
    #umb2=[min(Umbral2), max(Umbral2)]
    #umb2=[statistics.mode(Umbral2)]
    #Umbrales2.append(umb2)
    #umb3=[min(Umbral3), max(Umbral3)]
    #umb3=[statistics.mode(Umbral3)]
    #Umbrales3.append(umb3)
    #umb4=[min(Umbral4), max(Umbral4)]
    #umb4=[statistics.mode(Umbral4)]
    #Umbrales4.append(umb4)
    #umb5=[min(Umbral5), max(Umbral5)]
    #umb5=[statistics.mode(Umbral5)]
    #Umbrales5.append(umb5)
    # print(Umbral5)
    # print("\n\n")
    
    Data2,DiferenciaTotal,X_Diferencia_opinion=Entrenamiento(val_set.X_test, val_set.y_test)
    ValoracionOpinion=Evaluador(X_Diferencia_opinion)
    yreal=[]
    for id in val_set.y_test:
        yreal.append(id)
    print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
    pliegue+=1
    
# UmbralesFinales.append(Umbrales1)
# UmbralesFinales.append(Umbrales2)
# UmbralesFinales.append(Umbrales3)
# UmbralesFinales.append(Umbrales4)
# UmbralesFinales.append(Umbrales5)
# print(UmbralesFinales)

Data2,DiferenciaTotal,X_Diferencia_opinion=Entrenamiento(my_test_set.X_test, my_test_set.y_test)
ValoracionOpinion=Evaluador(X_Diferencia_opinion)

yreal=[]
for id in my_test_set.y_test:
        yreal.append(id)
print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
target_names=["1","2","3","4","5"]
print(classification_report(yreal, ValoracionOpinion, target_names=target_names))
print (confusion_matrix(yreal, ValoracionOpinion))
ConfusionMatrixDisplay.from_predictions(yreal, ValoracionOpinion)
plt.show()