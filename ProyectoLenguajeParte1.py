from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import pickle
import numpy as np
import statistics
import sklearn.metrics as met
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#------------------------------------Clases-----------------------------------------
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

#-----------------------------------Funciones---------------------------------------
def LematizadorOpiniones(opiniones,nlp):
    opiniones_lematizadas=[]
    for opinion in opiniones:
        if str(opinion)!=str("nan"):
            doc=nlp(opinion)
            opinion_tokenizada=[]
            opinion_lematizada=[]
            opinion_tokenizada=[token for token in doc]
            opinion_lematizada=[token.lemma_ for token in opinion_tokenizada]
            str_opinion_lematizada=""
            str_opinion_lematizada=" ".join(opinion_lematizada)
            opiniones_lematizadas.append(str_opinion_lematizada)
        else:
            opiniones_lematizadas.append(" ")
    return opiniones_lematizadas

def LematizadorTitulos(titles,nlp):
    titulos_lematizados=[]
    for titulo in titles:
        if str(titulo)!=str("nan"):
            doc=nlp(str(titulo))
            titulo_tokenizado=[]
            titulo_lematizado=[]
            titulo_tokenizado=[token for token in doc]
            titulo_lematizado=[token.lemma_ for token in titulo_tokenizado]
            str_titulo_lematizado=""
            str_titulo_lematizado=" ".join(titulo_lematizado)
            titulos_lematizados.append(str_titulo_lematizado)
        else: 
            titulos_lematizados.append(" ")
    return titulos_lematizados

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
                PalabraDesconocida+=1
                # print("No exite esa palabra en el diccionario")

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

def Evaluador(Diferencia, UmbralesFinales):
    i=0
    ValoracionOpinion=[]
    for df in Diferencia:
        if df <= UmbralesFinales[0]:
            ValoracionOpinion.append(1)
        elif  df <= UmbralesFinales[1]:
            ValoracionOpinion.append(2)
        elif  df <= UmbralesFinales[2]:
            ValoracionOpinion.append(3)
        elif  df <= UmbralesFinales[3]:
            ValoracionOpinion.append(4)
        elif df > UmbralesFinales[4]:
            ValoracionOpinion.append(5)
        else:
            ValoracionOpinion.append(2)
        # print("Valoracion Predicha Opinion["+str(i)+"]: "+str(ValoracionOpinion[len(ValoracionOpinion) - 1])+", valor real: "+str(y_train[i]))
        i+=1
    return ValoracionOpinion

#----------------------------------Inicio----------------------------------

#Esta parte solo es para tokenizar y lematizar el corpus y despues guardar los resultados en un pickle

# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')

# titles = df['Title'].values
# opinions = df['Opinion'].values
# polarity=df['Polarity'].values
# attraction=df['Attraction'].values

# #Cargamos corpus de spacy para tokenizar y lematizar
# nlp=spacy.load('es_core_news_sm')


# #Tokenizamos y lemmatizamos opiniones
# opiniones_lematizadas=LematizadorOpiniones(opinions,nlp)
# titulos_lematizados=LematizadorTitulos(titles,nlp)

# #Guardamos opiniones y titulos lematizados en pickle

# matriz_lematizados=[]
# headers=[("Title"),("Opinion"),("Polarity")]
# matriz_lematizados.append(headers)
# for titulo_lematizado,opinion_lematizada,polaridad in zip(titulos_lematizados,opiniones_lematizadas,polarity):
#     matriz_lematizados.append([titulo_lematizado,opinion_lematizada,polaridad])


# #Guardamos matriz_lematizados en pickle
# with open("matriz_opiniones_titulos_lematizados.pickle", "wb") as f:
#     pickle.dump(matriz_lematizados, f)

with open("matriz_opiniones_titulos_lematizados.pickle", "rb") as f:
    obj = pickle.load(f)
df=pd.DataFrame(obj)

# Praparación del diccionario de términos

# #Carga del archivo txt a un dataframe para poder almacenarlo y convertirlo en diccionario
# df_dic = pd.read_csv('SEL_full.txt', sep=str("\t"), header=None, names=['Palabra', 'Nula[%]','Baja[%]', 'Media[%]', 'Alta[%]', 'PFA', 'Categoría'], skiprows=1, encoding='utf-8')
# df_dic.drop(['Nula[%]', 'Baja[%]', 'Media[%]', 'Alta[%]'], axis=1, inplace=True)
# df_dic['PFA'].apply(lambda x: float(x))
# palabras_unicas=df_dic.Palabra.unique()

# valores=[]
# for palabra in palabras_unicas:
#     positivo=0.0
#     negativo=0.0
#     subdf= pd.DataFrame(df_dic[df_dic.Palabra==palabra])
#     for i in range(len(subdf)):
#         if str(subdf.iloc[i].Categoría)=="Alegría" or str(subdf.iloc[i].Categoría)=="Sorpresa":
#             positivo= positivo + subdf.iloc[i].PFA
#         elif str(subdf.iloc[i].Categoría)=="Enojo" or str(subdf.iloc[i].Categoría)=="Miedo" or str(subdf.iloc[i].Categoría)=="Repulsión" or str(subdf.iloc[i].Categoría)=="Tristeza":
#             negativo= negativo + subdf.iloc[i].PFA
#     valores.append([str(palabra), positivo, negativo])

# df_diccionary=pd.DataFrame(valores, columns=['Palabra', 'Positivo', 'Negativo'])

# #Convertimos a un diccionario
# diccionario = df_diccionary.set_index('Palabra').T.to_dict('list')

# #Guardamos el diccionario en un archivo .pkl
# with open("Dictionary.pickle", "wb") as tf:
#     pickle.dump(diccionario,tf)

with open("Dictionary.pickle", "rb") as f:
    dic = pickle.load(f)

# Pruebas del diccionario
# print(dic)
# for i in dic:
#     print(i)
# print(dic.get("alma"))

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

#-------------------------------------Pliegues--------------------------------------#
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

#-------------------------------------Entrenamiento--------------------------------------#

# Umbrales obtenidos luego de hacer muchos expereimentos (prueba y error)
UmbralesFinales=[-0.9, -0.4, -0.2, -0.1, -0.1]

pliegue=1
for val_set in my_data_set.validation_set:
    print("Pliegue: ",pliegue)
    
    #Datos de entrenamiento
    Data,DiferenciaTotal,X_Diferencia_opinion= Entrenamiento(val_set.X_train,val_set.y_train)
    ValoracionOpinion=Evaluador(X_Diferencia_opinion, UmbralesFinales)
    yreal=[]
    for id in val_set.y_train:
        yreal.append(id)
        
    # Métricas para medir la eficiencia de cada experimento en cada pliegue
    
    # print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
    # print('Precision:',precision_score(yreal, ValoracionOpinion, average='micro'))
    # print('Recall: ', recall_score(yreal, ValoracionOpinion, average='micro'))
    # print('F-Measure:', f1_score(yreal, ValoracionOpinion, average='micro'))
    # print("\n")
    
    Data2,DiferenciaTotal,X_Diferencia_opinion=Entrenamiento(val_set.X_test, val_set.y_test)
    ValoracionOpinion=Evaluador(X_Diferencia_opinion, UmbralesFinales)
    # yreal=[]
    # for id in val_set.y_test:
    #     yreal.append(id)
    # print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
    
    pliegue+=1
print("\n")

#-----------------------------Predicción en el conjunto de prueba----------------------------#
Data2,DiferenciaTotal,X_Diferencia_opinion=Entrenamiento(my_test_set.X_test, my_test_set.y_test)
ValoracionOpinion=Evaluador(X_Diferencia_opinion, UmbralesFinales)

yreal=[]
for id in my_test_set.y_test:
        yreal.append(id)
print("\n")
print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
# print('Accuracy:',met.accuracy_score(yreal, ValoracionOpinion)) 
# print('Precision:',precision_score(yreal, ValoracionOpinion, average='micro'))
# print('Recall: ', recall_score(yreal, ValoracionOpinion, average='micro'))
# print('F-Measure:', f1_score(yreal, ValoracionOpinion, average='micro'))
# print("\n")
target_names=["1","2","3","4","5"]
print(classification_report(yreal, ValoracionOpinion, target_names=target_names))
print (confusion_matrix(yreal, ValoracionOpinion))
ConfusionMatrixDisplay.from_predictions(yreal, ValoracionOpinion)
plt.show()
