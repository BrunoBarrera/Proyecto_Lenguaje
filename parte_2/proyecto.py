import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import spacy
import pickle
import sklearn.metrics as met
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import statistics
import csv
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#--------------------------FUNCIONES---------------------------
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

# #Pasamos polarity y attraction a lista
# polaridades=list(polarity)
# atracciones=list(attraction)

# #Guardamos opiniones y titulos lematizados,polaridades y atracciones en pickle
# matriz_OTPA=[]
# for titulo_lematizado,opinion_lematizada,polaridad,atraccion in zip(titulos_lematizados,opiniones_lematizadas,polaridades,atracciones):
#     matriz_OTPA.append([titulo_lematizado,opinion_lematizada,polaridad,atraccion])


# #Guardamos matriz_OTA en pickle
# with open("matriz_opiniones_titulos_polaridades_atracciones.pickle", "wb") as f:
#     pickle.dump(matriz_OTPA, f)  


with open("matriz_opiniones_titulos_polaridades_atracciones.pickle", "rb") as f:
    obj = pickle.load(f)


#Guardamos los datos de entrenamiento en X
X=[]
for i in range(len(obj)):
    if i!=0:
        x1=str(obj[i][0])
        x2=str(obj[i][1])
        X.append([x1.split(),x2.split()])

# Guardamos las etiquetas en y para polaridad
y_polaridad=[]
for i in range(len(obj)):
    if i!=0:
        y_polaridad.append(obj[i][2])


# Guardamos las etiquetas en y para atraccion
y_atraccion=[]
for i in range(len(obj)):
    if i!=0:
        y_atraccion.append(obj[i][3])

#Dividimos en train y test

#Aqui no ocupo como tal "X_train_atraccion" ni "X_test_atraccion" por que tienen lo mismo que "X_train_polaridad" Y "X_test_polaridad", 
#pero necesitaba guardar por separado las "y" de atraccion y las "y" de polaridad


X_train_polaridad, X_test_polaridad, y_train_polaridad, y_test_polaridad= train_test_split(X, y_polaridad, test_size=0.2,random_state=0,shuffle= True)
X_train_atraccion, X_test_atraccion, y_train_atraccion, y_test_atraccion= train_test_split(X, y_atraccion, test_size=0.2,random_state=0,shuffle= True)

y_train_polaridad=np.array(y_train_polaridad)
y_test_polaridad=np.array(y_test_polaridad)
y_train_atraccion=np.array(y_train_atraccion)
y_test_atraccion=np.array(y_test_atraccion)


df_train=pd.DataFrame(X_train_polaridad) 

titulos_tmp=df_train.iloc[:, 0]
titulos_train=[]
for t in titulos_tmp:
    r=" ".join(t)
    titulos_train.append(r)

opiniones_tmp=df_train.iloc[:, 1]
opiniones_train=[]
for o in opiniones_tmp:
    r=" ".join(o)
    opiniones_train.append(r)

titulos_opiniones_train=[]
for t,o in zip(titulos_train,opiniones_train):
    titulos_opiniones_train.append(t+" "+o)




df_test=pd.DataFrame(X_test_polaridad) 

titulos_tmp=df_test.iloc[:, 0]
titulos_test=[]
for t in titulos_tmp:
    r=" ".join(t)
    titulos_test.append(r)

opiniones_tmp=df_test.iloc[:, 1]
opiniones_test=[]
for o in opiniones_tmp:
    r=" ".join(o)
    opiniones_test.append(r)

titulos_opiniones_test=[]
for t,o in zip(titulos_test,opiniones_test):
    titulos_opiniones_test.append(t+" "+o)



#------------------------------REPRESENTACION BINARIA TRAIN-----------------------------------------
vectorizador_binario = CountVectorizer(binary=True)

#------------CAMBIAR ESTA LINEA PARA SOLUCIONAR LO DEL TAMAÑO DEL PICKLE--------------------
X_titulos_opiniones_binarizada_train = vectorizador_binario.fit_transform(titulos_opiniones_train)

# X_titulos_opiniones_binarizada_train=X_titulos_opiniones_binarizada_train.toarray()

#----------------------------REPRESENTACION FRECUENCIA TRAIN-----------------------------------------
vectorizador_frecuencia = CountVectorizer(decode_error='ignore',strip_accents='unicode')
X_titulos_opiniones_frecuencia_train = vectorizador_frecuencia.fit_transform(titulos_opiniones_train)
# X_titulos_opiniones_frecuencia_train=X_titulos_opiniones_frecuencia_train.toarray()


#------------------------------REPRESENTACION BINARIA TEST----------------------------------------
vectorizador_binario = CountVectorizer(binary=True)

#------------CAMBIAR ESTA LINEA PARA SOLUCIONAR LO DEL TAMAÑO DEL PICKLE--------------------
X_titulos_opiniones_binarizada_test = vectorizador_binario.fit_transform(titulos_opiniones_test)
# X_titulos_opiniones_binarizada_test=X_titulos_opiniones_binarizada_test.toarray()

#----------------------------REPRESENTACION FRECUENCIA TEST-----------------------------------------
vectorizador_frecuencia = CountVectorizer(decode_error='ignore',strip_accents='unicode')
X_titulos_opiniones_frecuencia_test = vectorizador_frecuencia.fit_transform(titulos_opiniones_test)
# X_titulos_opiniones_frecuencia_test=X_titulos_opiniones_frecuencia_test.toarray()


# with open("X_titulos_opiniones_binarizada_train.pickle", "wb") as f:
#     pickle.dump(X_titulos_opiniones_binarizada_train, f) 

# with open("X_titulos_opiniones_frecuencia_train.pickle", "wb") as f:
#     pickle.dump(X_titulos_opiniones_frecuencia_train, f) 

# with open("X_titulos_opiniones_frecuencia_test.pickle", "wb") as f:
#     pickle.dump(X_titulos_opiniones_frecuencia_test, f) 

# with open("X_titulos_opiniones_binarizada_test.pickle", "wb") as f:
#     pickle.dump(X_titulos_opiniones_binarizada_test, f) 

# with open("y_train_polaridad.pickle", "wb") as f:
#     pickle.dump(y_train_polaridad, f) 

# with open("y_train_atraccion.pickle", "wb") as f:
#     pickle.dump(y_train_atraccion, f) 

# with open("y_test_polaridad.pickle", "wb") as f:
#     pickle.dump(y_test_polaridad, f) 

# with open("y_test_atraccion.pickle", "wb") as f:
#     pickle.dump(y_test_atraccion, f) 





# #----------------------------------AQUI EMPIEZAN LOS ENTRENAMIENTOS------------------

# with open("X_titulos_opiniones_binarizada_train.pickle", "rb") as f:
#     obj = pickle.load(f)
# #X_titulos_opiniones_binarizada_train=np.array(obj)
# X_titulos_opiniones_binarizada_train=obj


# # with open("X_titulos_opiniones_frecuencia_train.pickle", "rb") as f:
# #     obj = pickle.load(f)
# # X_titulos_opiniones_frecuencia_train=np.array(obj)


# with open("y_train_polaridad.pickle", "rb") as f:
#     obj = pickle.load(f)
# y_train_polaridad=np.array(obj)


# with open("y_train_atraccion.pickle", "rb") as f:
#     obj = pickle.load(f)
# y_train_atraccion=np.array(obj)


# #-------------------------------------Pliegues--------------------------------------#
#Validacion de 5 plieges
kf = KFold(n_splits=5)

X_train_binarizada_v=[]
y_train_polaridad_binarizada_v=[]
X_test_binarizada_v=[]
y_test_polaridad_binarizada_v=[]

y_train_atraccion_binarizada_v=[]
y_test_atraccion_binarizada_v=[]

for i_train, i_test in kf.split(X_titulos_opiniones_binarizada_train):
    X_train_binarizada_v.append(X_titulos_opiniones_binarizada_train[i_train])
    y_train_polaridad_binarizada_v.append(y_train_polaridad[i_train])
    X_test_binarizada_v.append(X_titulos_opiniones_binarizada_train[i_test])
    y_test_polaridad_binarizada_v.append(y_train_polaridad[i_test])

    y_train_atraccion_binarizada_v.append(y_train_polaridad[i_train])
    y_test_atraccion_binarizada_v.append(y_train_polaridad[i_test])


X_train_frecuencia_v=[]
y_train_polaridad_frecuencia_v=[]
X_test_frecuencia_v=[]
y_test_polaridad_frecuencia_v=[]

y_train_atraccion_frecuencia_v=[]
y_test_atraccion_frecuencia_v=[]

for i_train, i_test in kf.split(X_titulos_opiniones_frecuencia_train):
    X_train_frecuencia_v.append(X_titulos_opiniones_frecuencia_train[i_train])
    y_train_polaridad_frecuencia_v.append(y_train_polaridad[i_train])
    X_test_frecuencia_v.append(X_titulos_opiniones_frecuencia_train[i_test])
    y_test_polaridad_frecuencia_v.append(y_train_polaridad[i_test])

    y_train_atraccion_frecuencia_v.append(y_train_polaridad[i_train])
    y_test_atraccion_frecuencia_v.append(y_train_polaridad[i_test])



# print("---------------------------REPRESENTACION BINARIZADA---------------------------------\n")

#Empezamos con la polaridad

clf = LogisticRegression(random_state=0,max_iter=1000,C=.84)

promedio_accuracy_polaridad=[]
promedio_precision_polaridad=[]
promedio_recall_polaridad=[]
promedio_F_Measure_polaridad=[]

print("---------------------------REGRESION LOGISTICA---------------------------------------\n")
pliegue=0
print("---------------------------------POLARIDAD-------------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_binarizada_v,y_train_polaridad_binarizada_v,X_test_binarizada_v,y_test_polaridad_binarizada_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_polaridad.append(accuracy)
    promedio_precision_polaridad.append(precision)
    promedio_recall_polaridad.append(recall)
    promedio_F_Measure_polaridad.append(F_Measure)


    print("Accuracy Polaridad en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Polaridad en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Polaridad en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Polaridad en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("Promedio Accuracy Polaridad:      {}".format(statistics.mean(promedio_accuracy_polaridad)))
print("Promedio Precision Polaridad:     {}".format(statistics.mean(promedio_precision_polaridad)))
print("Promedio Recall Polaridad:        {}".format(statistics.mean(promedio_recall_polaridad)))
print("Promedio F-Measure Polaridad:     {}".format(statistics.mean(promedio_F_Measure_polaridad)))
print("\n")
#Ahora con la atraccion
pliegue=0
promedio_accuracy_atraccion=[]
promedio_precision_atraccion=[]
promedio_recall_atraccion=[]
promedio_F_Measure_atraccion=[]
pliegue=0
print("---------------------------ATRACCION---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_binarizada_v,y_train_atraccion_binarizada_v,X_test_binarizada_v,y_test_atraccion_binarizada_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_atraccion.append(accuracy)
    promedio_precision_atraccion.append(precision)
    promedio_recall_atraccion.append(recall)
    promedio_F_Measure_atraccion.append(F_Measure)

    print("Accuracy Atraccion en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Atraccion en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Atraccion en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Atraccion en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Atraccion:     {}".format(statistics.mean(promedio_accuracy_atraccion)))
print("Promedio Precision Atraccion:    {}".format(statistics.mean(promedio_precision_atraccion)))
print("Promedio Recall Atraccion:       {}".format(statistics.mean(promedio_recall_atraccion)))
print("Promedio F-Measure Atraccion:    {}".format(statistics.mean(promedio_F_Measure_atraccion)))
print("\n")












#-----Entrenamos el modelo de Naive Bayes con los pliegues---------------------


clf = MultinomialNB()

promedio_accuracy_polaridad=[]
promedio_precision_polaridad=[]
promedio_recall_polaridad=[]
promedio_F_Measure_polaridad=[]

print("---------------------------NAIVE BAYES---------------------------------------\n")
pliegue=0
print("---------------------------POLARIDAD---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_binarizada_v,y_train_polaridad_binarizada_v,X_test_binarizada_v,y_test_polaridad_binarizada_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_polaridad.append(accuracy)
    promedio_precision_polaridad.append(precision)
    promedio_recall_polaridad.append(recall)
    promedio_F_Measure_polaridad.append(F_Measure)


    print("Accuracy Polaridad en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Polaridad en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Polaridad en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Polaridad en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Polaridad:      {}".format(statistics.mean(promedio_accuracy_polaridad)))
print("Promedio Precision Polaridad:     {}".format(statistics.mean(promedio_precision_polaridad)))
print("Promedio Recall Polaridad:        {}".format(statistics.mean(promedio_recall_polaridad)))
print("Promedio F-Measure Polaridad:     {}".format(statistics.mean(promedio_F_Measure_polaridad)))
print("\n")
#Ahora con la atraccion

promedio_accuracy_atraccion=[]
promedio_precision_atraccion=[]
promedio_recall_atraccion=[]
promedio_F_Measure_atraccion=[]

print("---------------------------ATRACCION---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_binarizada_v,y_train_atraccion_binarizada_v,X_test_binarizada_v,y_test_atraccion_binarizada_v):

    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_atraccion.append(accuracy)
    promedio_precision_atraccion.append(precision)
    promedio_recall_atraccion.append(recall)
    promedio_F_Measure_atraccion.append(F_Measure)

    print("Accuracy Atraccion en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Atraccion en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Atraccion en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Atraccion en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Atraccion:     {}".format(statistics.mean(promedio_accuracy_atraccion)))
print("Promedio Precision Atraccion:    {}".format(statistics.mean(promedio_precision_atraccion)))
print("Promedio Recall Atraccion:       {}".format(statistics.mean(promedio_recall_atraccion)))
print("Promedio F-Measure Atraccion:    {}".format(statistics.mean(promedio_F_Measure_atraccion)))







#-----------------------------REPRESENTACION FRECUENCIA-----------------------------------

clf = LogisticRegression(random_state=0,max_iter=900)
promedio_accuracy_polaridad=[]
promedio_precision_polaridad=[]
promedio_recall_polaridad=[]
promedio_F_Measure_polaridad=[]

print("---------------------------REPRESENTACION FRECUENCIA---------------------------------\n")
print("---------------------------REGRESION LOGISTICA---------------------------------------\n")
pliegue=0
print("---------------------------------POLARIDAD-------------------------------------------\n")

for x_train,y_train,x_test,y_test in zip(X_train_frecuencia_v,y_train_polaridad_frecuencia_v,X_test_frecuencia_v,y_test_polaridad_frecuencia_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_polaridad.append(accuracy)
    promedio_precision_polaridad.append(precision)
    promedio_recall_polaridad.append(recall)
    promedio_F_Measure_polaridad.append(F_Measure)


    print("Accuracy Polaridad en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Polaridad en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Polaridad en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Polaridad en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("Promedio Accuracy Polaridad:      {}".format(statistics.mean(promedio_accuracy_polaridad)))
print("Promedio Precision Polaridad:     {}".format(statistics.mean(promedio_precision_polaridad)))
print("Promedio Recall Polaridad:        {}".format(statistics.mean(promedio_recall_polaridad)))
print("Promedio F-Measure Polaridad:     {}".format(statistics.mean(promedio_F_Measure_polaridad)))
print("\n")
#Ahora con la atraccion
pliegue=0
promedio_accuracy_atraccion=[]
promedio_precision_atraccion=[]
promedio_recall_atraccion=[]
promedio_F_Measure_atraccion=[]
pliegue=0


promedio_accuracy_polaridad=[]
promedio_precision_polaridad=[]
promedio_recall_polaridad=[]
promedio_F_Measure_polaridad=[]

print("---------------------------ATRACCION---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_frecuencia_v,y_train_atraccion_frecuencia_v,X_test_frecuencia_v,y_test_atraccion_frecuencia_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_atraccion.append(accuracy)
    promedio_precision_atraccion.append(precision)
    promedio_recall_atraccion.append(recall)
    promedio_F_Measure_atraccion.append(F_Measure)

    print("Accuracy Atraccion en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Atraccion en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Atraccion en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Atraccion en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Atraccion:     {}".format(statistics.mean(promedio_accuracy_atraccion)))
print("Promedio Precision Atraccion:    {}".format(statistics.mean(promedio_precision_atraccion)))
print("Promedio Recall Atraccion:       {}".format(statistics.mean(promedio_recall_atraccion)))
print("Promedio F-Measure Atraccion:    {}".format(statistics.mean(promedio_F_Measure_atraccion)))
print("\n")







#-----Entrenamos el modelo de Naive Bayes con los pliegues---------------------


clf = MultinomialNB()
promedio_accuracy_polaridad=[]
promedio_precision_polaridad=[]
promedio_recall_polaridad=[]
promedio_F_Measure_polaridad=[]



print("---------------------------NAIVE BAYES---------------------------------------\n")
pliegue=0
print("---------------------------POLARIDAD---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_frecuencia_v,y_train_polaridad_frecuencia_v,X_test_frecuencia_v,y_test_polaridad_frecuencia_v):
    pliegue+=1
    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_polaridad.append(accuracy)
    promedio_precision_polaridad.append(precision)
    promedio_recall_polaridad.append(recall)
    promedio_F_Measure_polaridad.append(F_Measure)


    print("Accuracy Polaridad en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Polaridad en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Polaridad en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Polaridad en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Polaridad:      {}".format(statistics.mean(promedio_accuracy_polaridad)))
print("Promedio Precision Polaridad:     {}".format(statistics.mean(promedio_precision_polaridad)))
print("Promedio Recall Polaridad:        {}".format(statistics.mean(promedio_recall_polaridad)))
print("Promedio F-Measure Polaridad:     {}".format(statistics.mean(promedio_F_Measure_polaridad)))
print("\n")
#Ahora con la atraccion

promedio_accuracy_atraccion=[]
promedio_precision_atraccion=[]
promedio_recall_atraccion=[]
promedio_F_Measure_atraccion=[]

print("---------------------------ATRACCION---------------------------------------\n")
for x_train,y_train,x_test,y_test in zip(X_train_frecuencia_v,y_train_atraccion_frecuencia_v,X_test_frecuencia_v,y_test_atraccion_frecuencia_v):

    clf.fit(x_train,y_train)
    prediccion=clf.predict(x_test)

    accuracy=met.accuracy_score(y_test, prediccion)
    precision=precision_score(y_test, prediccion, average='micro')
    recall=recall_score(y_test, prediccion, average='micro')
    F_Measure=f1_score(y_test, prediccion, average='micro')

    promedio_accuracy_atraccion.append(accuracy)
    promedio_precision_atraccion.append(precision)
    promedio_recall_atraccion.append(recall)
    promedio_F_Measure_atraccion.append(F_Measure)

    print("Accuracy Atraccion en pliegue {}:     {}".format(pliegue,accuracy))
    print("Precision Atraccion en pliegue {}:    {}".format(pliegue,precision))
    print("Recall Atraccion en pliegue {}:       {}".format(pliegue,recall))
    print("F-Measure Atraccion en pliegue {}:    {}".format(pliegue,F_Measure))
    print("\n")
print("\nPromedio Accuracy Atraccion:     {}".format(statistics.mean(promedio_accuracy_atraccion)))
print("Promedio Precision Atraccion:    {}".format(statistics.mean(promedio_precision_atraccion)))
print("Promedio Recall Atraccion:       {}".format(statistics.mean(promedio_recall_atraccion)))
print("Promedio F-Measure Atraccion:    {}".format(statistics.mean(promedio_F_Measure_atraccion)))