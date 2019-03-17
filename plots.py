#GRÁFICAS A REPRESENTAR EN EL TFM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##################################################################
# UNIQUES PLOT
##################################################################

def uniquesPlot(df, var=False, title=None, v_diff = [''], threshold_min=0, threshold_max=1, path_dest='./', figname='fig', figsize=(5, 8.27), save_name=None):
    #df: dataframe con las variables
    #var: variables específicas, por defecto coge todas las que contienen nulos
    #v_diff: variables que deben ser representadas de manera diferente
    #threshold_min: umbral mínimo para mostrar únicos
    #threshold_max: umbral máximo para mostrar únicos
    #path_dest: ruta donde guardar la imagen
    #figsize: tamaño de la imagen
    #save_name: nombre del archivo a guardar
    
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    
    if var is False:
        var = df.columns

    #Rango de valores a mostrar
    tuplas_pares = [(x, len(list(df[x].unique()))) for x in var if len(list(df[x].unique()))/df.shape[0] >= threshold_min and len(list(df[x].unique()))/df.shape[0] <= threshold_max]

    tuplas_pares = sorted(tuplas_pares, key=lambda x: x[1], reverse=True)
    v_labels = [x[0] for x in tuplas_pares]
    v_size = [x[1] for x in tuplas_pares]
    
    barh1 = ax1.barh(v_labels, v_size, color='red', alpha=1)
    
    #barras que deben ser coloreadas de diferente manera:
    ind_dif = [i for i,x in enumerate(tuplas_pares) if x[0] in v_diff]
    for i in ind_dif:
        barh1[i].set_color('red')
        barh1[i].set_alpha(0.4)
    
    for rect in barh1:
        width = rect.get_width()
        ax1.text(width, rect.get_y() + rect.get_height()/2.0, '{:.2%}'.format(width/df.shape[0]), va='center', ha='left', rotation=0)
    
    if title:
        plt.title(figname, fontsize=16)

    if save_name:
        plt.savefig(path_dest+save_name+'.png')
    else:
        plt.show()

        

        
##################################################################
# NULL PLOTS
##################################################################

def nanPlot(df, var=None, v_diff = [''], title=None, path_dest='./', figname='fig', figsize=(5, 8.27), save_name=None):
    #df: dataframe con las variables
    #var: variables específicas, por defecto coge todas las que contienen nulos
    #v_diff: variables que deben ser representadas de manera diferente
    #path_dest: ruta donde guardar la imagen
    #figsize: tamaño de la imagen
    #save_name: nombre del archivo a guardar
    
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    
    if not var:
        var = list(df.columns)
    
    tuplas_pares = [(x, df[x].isnull().sum()) for x in var if df[x].isnull().sum() > 0]
    tuplas_pares = sorted(tuplas_pares, key=lambda x: x[1], reverse=True)
    v_labels = [x[0] for x in tuplas_pares]
    v_size = [x[1] for x in tuplas_pares]
    
    barh1 = ax1.barh(v_labels, v_size, color='red', alpha=1)
    
    #barras que deben ser coloreadas de diferente manera:
    ind_dif = [i for i,x in enumerate(tuplas_pares) if x[0] in v_diff]
    for i in ind_dif:
        barh1[i].set_color('red')
        barh1[i].set_alpha(0.4)
    
    for rect in barh1:
        width = rect.get_width()
        ax1.text(width, rect.get_y() + rect.get_height()/2.0, '{:.2%}'.format(width/df.shape[0]), va='center', ha='left', rotation=0)
    
    if title:
        plt.title(figname, fontsize=16)

    if save_name:
        plt.savefig(path_dest+save_name+'.png')
    else:
        plt.show()
        
        
        
##################################################################
# NUMERIC PLOT
##################################################################   

def numericPlot(v, path_dest='./', figname='fig', figsize=(30, 10), save=None):
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 5), (0, 2))
    ax3 = plt.subplot2grid((1, 5), (0, 3))
    ax4 = plt.subplot2grid((1, 5), (0, 4))
    
    ax1.bar(v.value_counts().index.tolist(), v.value_counts()) #Histograma bruto por valor
    ax2.hist(v)
    ax3.violinplot(v, vert=False)
    ax4.boxplot(v, vert=False)
    
    fig.suptitle(figname, fontsize=16)

    if save:
        plt.savefig(path_dest+figname+'.png')
    else:
        plt.show()
        
        

        
##################################################################
# NOMINAL PLOT
##################################################################   

def nominalPlot(v, path_dest='./', figname='fig', figsize=(15, 15), save=True):
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=5)
    
    ax1.bar(v.value_counts().index.tolist(), v.value_counts()) #Histograma bruto por valor
    ax1.tick_params(axis='x', labelrotation=90)
    fig.suptitle(figname, fontsize=16)

    if save:
        plt.savefig(path_dest+figname+'.png')
    else:
        plt.show()
        

        
        
##################################################################
# BAR PLOT
##################################################################         
        
def plot_bar(y1, num_format1=None, num_format2=None, ylabel=None, xlabel=None, ylim=1, title=None, subtitle=None, figsize=(8.27,5), opacity = 0.75, bar_width = 0.8, 
             color1 = 'red', save_nomvar=None, path_output='./'):
    
    # ELEMENTOS DE LA FUNCIÓN
    #y1: vector con los datos a representar
    #y2_names=False: 2 vector con valores diferentes a representar encima de las barras
    #x_names=False: vector con nombres del eje x
    #title='Title': Título
    #subtitle=False: Subtítulo
    #x_size = 15: Ancho del gráfico
    #opacity = 0.75: transparencia de los datos representados
    #bar_width = 0.4: ancho de la barra
    #num_format1 = '{:.1f}': formato de los valores de y1 (float por defecto)
    #num_format2 = '{:,.1%}': formato de los valores de y2 (porcentaje por defecto)
    #color1 = 'red': color de y1
    #save_nomvar=None: guardar gráfico si se incluye un nombre
    #path_output = './': ruta donde guardar los gráficos

    
    #1. Creando el gráfico base
    fig, ax = plt.subplots(figsize=figsize)
    bar1 = ax.bar(y1.value_counts().index.tolist(), y1.value_counts()/len(y1), bar_width, align='center', alpha=opacity, color=color1)
            
    #2. Añadiendo numeros en la altura media de las barras
    if num_format1:
        for h_1 in bar1:
                height = h_1.get_height()            
                ax.text(h_1.get_x() + h_1.get_width()/2.0, height/2-height*0.03, num_format1.format(height), ha='center', va='bottom', color='white',weight='bold')
            
    #3. Incluyendo el número absoluto encima de la barra
    if num_format2:
        for h_name, h_1 in zip(y1.value_counts(), bar1):
            height = h_1.get_height()
            ax.text(h_1.get_x() + h_1.get_width()/2.0, height, num_format2.format(h_name), ha='center', va='bottom', color='dimgray', weight='bold')

    #4. Ajustando parámetros del gráfico: título, subtitulo, límites y ticks 
    if title:
        ax.set_title(title, weight='bold')
        
    if ylabel:
        plt.ylabel(ylabel, weight='bold')
    
    if xlabel:
        plt.xlabel(xlabel, weight='bold')
        
    ax.set_ylim(0, ylim) #Con esto controlamos la altura de las gráficas
    if isinstance(y1[0], str):
        ax.tick_params(axis='x', labelrotation=90)
    
    #6. Guardando gráfica si se da un nombre
    if save_nomvar:
        plt.savefig(path_output+save_nomvar+'.png', dpi = 400, transparent=True)
    
    #7. Mostrar por pantalla la gráfica
    plt.show()

    

##################################################################
# HIST PLOT
##################################################################  
def plot_hist(v, title=None, ylabel=None, xlabel=None, figsize=(8.27,5), opacity = 0.75, save_nomvar=None, path_output='./'):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1.hist(v, color='red', alpha=opacity)
    
    if title:
        plt.title(title, weight='bold')
    
    if ylabel:
        plt.ylabel(ylabel, weight='bold')
    
    if xlabel:
        plt.xlabel(xlabel, weight='bold')

    if save_nomvar:
        plt.savefig(path_output+save_nomvar+'.png')
    
    plt.show()   
    

##################################################################
# HIST PLOT BREAK
################################################################## 

def plot_hist_break(v1, v2, title=None, ylabel=None, xlabel=None, figsize=(8.27,5), opacity = 0.75, save_nomvar=None, path_output='./'):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    
    ax1.hist(v1, color='red', alpha=opacity)
    ax2.hist(v2, color='red', alpha=opacity)
    
       

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    #ax1.tick_params(labelright='off')
    #ax2.yaxis.tick_right()
    
    if title:
        fig.suptitle(title, weight='bold')
    
    
    # Añadir rayas diagonales
        
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'
    
    
    if ylabel:
        plt.ylabel(ylabel, weight='bold')
    
    if xlabel:
        plt.xlabel(xlabel, weight='bold')
    
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    
    
    
    if save_nomvar:
        plt.savefig(path_output+save_nomvar+'.png')
    
    plt.show()       
    
    
##################################################################
# BOX PLOT
################################################################## 
def plot_boxplot(v, title=None, ylabel=None, xlabel=None, figsize=(5, 8.27), opacity = 0.75, save_nomvar=None, path_output='./'):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1.boxplot(v, vert=True)
    
    if title:
        plt.title(title, weight='bold')
    
    if ylabel:
        plt.ylabel(ylabel, weight='bold')
    
    if xlabel:
        plt.xlabel(xlabel, weight='bold')

    if save_nomvar:
        plt.savefig(path_output+save_nomvar+'.png')
    
    plt.show() 
    
    
    
    
    
##################################################################
# CORRELATION PLOT
##################################################################  

def plot_corr(df, method='pearson', figsize=(20,20), path_dest='./', save_name='corrplot'):
    import matplotlib.pyplot as plt
    
    #annot sirve para añadir los numeritos
    plt.figure(figsize=figsize)
    corr = df.corr(method=method) #pearson, kendall, spearman
    ax = sns.heatmap(corr, square = True, cmap="Reds", vmax=1, robust=True, linewidths=0.5, linecolor = 'k',  cbar_kws={"use_gridspec":False, "location":"top", "shrink": .60}, annot=False, annot_kws={"weight": "bold"}) #If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, weight='bold')
    fig_corrplot = ax.get_figure()
    fig_corrplot.savefig(path_dest+save_name+".png")
