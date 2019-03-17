# FUNCIONES A UTILIZAR EN EL TFM
import pandas as pd
import numpy as np



##################################################################
# ENCODER BY FREQUENCY
##################################################################

def freq_encoding(df, var, offset=1, float_freq=False):
    # Ordena las clases de mayor a menor frecuencia
    #  float_freq: si es True entonces pone el valor de la frecuencia en número real
    # HACERLO DESDE 1 PARA PODER UTILIZAR 0 COMO NAs EN MODELOS LINEALES REGRESIVOS
    
    if float_freq is True:
        tuplas_pares = [(x,y) for x,y in zip(df[var].value_counts().index.tolist(), df[var].value_counts()/df[var].shape[0])] #indice, frecuencia en double
    else:
        tuplas_pares = [(x,y) for x,y in zip(df[var].value_counts().index.tolist(), df[var].value_counts())] #indice, frecuencia en int
    tuplas_pares = sorted(tuplas_pares, key=lambda x: x[1], reverse=True)
    print(tuplas_pares)
    
    df2 = df.copy() #hay que crear obligatoriamente un vector copia porque si el valor cambia a un numero es posible que en otra vuelta del bucle ese numero lo cambie
    for i,x in enumerate(tuplas_pares):
            ind = df2.index[df2[var] == x[0]].tolist() #Saco los indices de aquellas clases de la variable que sean x[0]
            df.at[ind, var] = i+offset #Los índices que cumplan la condición en df2 se cambian en df. Los cambios son los nuevos indices
            #print(df[var].unique())
            
            

##################################################################
# GROUP UNDERCLASSES
##################################################################

def group_underclasses(df, var, threshold=0.05, underclass_name = None):

    
    #Se buscan índices de las clases que tienen una frecuencia menor al umbral
    v=df[var]
    res = {x:y for x,y in zip(v.value_counts().index.tolist(), v.value_counts()/len(v)) if y <= threshold}
    
    if res:
        if not underclass_name:
            underclass_name = list(res.keys())[0]

        #Se cambian los que cumplan la condición
        for x in res.keys():
            df.loc[df[var] == x, var] = underclass_name
