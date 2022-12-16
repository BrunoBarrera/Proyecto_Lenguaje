import pandas as pd
import spacy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met
from sklearn.linear_model import LogisticRegression

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
    
# print(dic)
# for i in dic:
#     print(i)
# print(dic.get("alma"))

pliege=1
Umbral=[]

#Datos de entrenamiento
i=0
ValoracionOpinion=[]
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
        # Data.append([opinion_positivo,opinion_negativo])
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


# plt.plot(x,y,"o")
# plt.show()
kmeans = KMeans(n_clusters= 5)
label = kmeans.fit_predict(Data)
clus=[]
for x,y in zip(Data,label):
    clus.append([x,y])
    

#Getting unique labels
 
u_labels = np.unique(label)
centroids = kmeans.cluster_centers_
# print(centroids)

# for i in u_labels:
#     x=[]
#     y=[]
#     for point in clus:
#         if point[1]==i:
#             x.append(point[0][0])
#             y.append(point[0][1])
#     plt.scatter(x , y , label = i)
# plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
# plt.legend()
# plt.show()

#Getting the Centroids
for cen in centroids:
    Umbral.append(cen[0]-cen[1])
Umbral.sort()
print(Umbral)
#plotting the results
# plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
# plt.show()
# kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(Data)
    #Prediccion
    #val_set.y_train
    #Entrenamiento algoritmo optimizacion 
    
    #Valuacion de algoritmo de optimizacion
        #val_set.X_test
        #val_set.y_test

        #Asignar Valoracion por umbral
pliege+=1
# Umbral=[-0.5,-0.2,0,0.2,0.5]
clf = LogisticRegression()
clf.fit(Data, y_train)
Data2=[]
DiferenciaTotal=[]
i=0
for opinion in X_test:
        PalabraDesconocida=0
        titulo_Positivo=0
        titulo_Negativo=0
        opinion_positivo=0
        opinion_negativo=0
        for word in opinion[0]:
            try:
                Sentimiento=dic.get(word)[0]
                titulo_Positivo+=dic.get(word)[0]
                titulo_Negativo+=dic.get(word)[1]
            except:
                PalabraDesconocida+=1
        for word in opinion[1]:
            try:
                Sentimiento=dic.get(word)[1]
                opinion_positivo+=dic.get(word)[0]
                opinion_negativo+=dic.get(word)[1]
            except:
                PalabraDesconocida=PalabraDesconocida
        Data2.append([opinion_positivo,opinion_negativo])
        DiferenciaTotal.append(opinion_positivo-opinion_negativo)
      
        i+=1
y_pred = clf.predict(Data2) 
i=0
for df in DiferenciaTotal:
    if df <= Umbral[0]:
        ValoracionOpinion.append(5)
    elif  df <= Umbral[1]:
        ValoracionOpinion.append(4)
    elif  df <= Umbral[2]:
        ValoracionOpinion.append(3)
    elif  df <= Umbral[3]:
        ValoracionOpinion.append(2)
    # elif  df >= Umbral[4]:
    else:
        ValoracionOpinion.append(1)
    # print("Valoracion Predicha Opinion["+str(i)+"]: "+str(ValoracionOpinion[len(ValoracionOpinion) - 1])+", valor real: "+str(y_train[i]))
    i+=1
print('Accuracy:',met.accuracy_score(y_test, ValoracionOpinion)) 
print('Accuracy Logistic:',met.accuracy_score(y_test, y_pred)) 

