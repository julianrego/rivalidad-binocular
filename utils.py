import numpy as np
from scipy.integrate import odeint
import pandas as pd

def get_dominance_times(estimulo_derecho, estimulo_izquierdo, t):

    df_r = pd.DataFrame(data={'E':estimulo_derecho}, index = t)
    df_l = pd.DataFrame(data={'E':estimulo_izquierdo}, index = t)


    df_r['threshold'] =(df_r['E']>0.05).astype(int) 
    df_r['dif'] = df_r['threshold'].shift(-1) - df_r['threshold'] 

    df_l['threshold'] =(df_l['E']>0.05).astype(int) 
    df_l['dif'] = df_l['threshold'].shift(-1) - df_l['threshold'] 

    #ojo derecho
    tiempos_arriba_der = []
    tiempos_abajo_der = []
    for idx, e in df_r.iterrows():

        if e.dif==1:
            tiempos_arriba_der.append(idx)
        if e.dif==-1:
            tiempos_abajo_der.append(idx)
        
    tiempos_arriba_der = np.array(tiempos_arriba_der)
    tiempos_abajo_der = np.array(tiempos_abajo_der)

    min_len = min(len(tiempos_arriba_der), len(tiempos_abajo_der))
    derecho = tiempos_abajo_der[:min_len-1] - tiempos_arriba_der[:min_len-1]

    #ojo izquierdo
    tiempos_arriba_izq = []
    tiempos_abajo_izq = []
    for idx, e in df_l.iterrows():

        if e.dif==1:
            tiempos_arriba_izq.append(idx)
        if e.dif==-1:
            tiempos_abajo_izq.append(idx)
        
    tiempos_arriba_izq = np.array(tiempos_arriba_izq)
    tiempos_abajo_izq = np.array(tiempos_abajo_izq)

    min_len = min(len(tiempos_arriba_izq), len(tiempos_abajo_izq))
    izquierdo =  tiempos_abajo_izq[:min_len-1] - tiempos_arriba_izq[:min_len-1]

    #promedios excluyendo los bordes

    derecho_mean = derecho[1:-1].mean()
    izquierdo_mean = izquierdo[1:-1].mean()

    results = {
        'derecho':derecho_mean,
        'izquierdo':izquierdo_mean
    }

    return results

def get_dominance_distribution(estimulo_derecho, estimulo_izquierdo, t):

    df_r = pd.DataFrame(data={'E':estimulo_derecho}, index = t)
    df_l = pd.DataFrame(data={'E':estimulo_izquierdo}, index = t)


    df_r['threshold'] =(df_r['E']>0.05).astype(int) 
    df_r['dif'] = df_r['threshold'].shift(-1) - df_r['threshold'] 

    df_l['threshold'] =(df_l['E']>0.05).astype(int) 
    df_l['dif'] = df_l['threshold'].shift(-1) - df_l['threshold'] 

    #ojo derecho
    tiempos_arriba_der = []
    tiempos_abajo_der = []
    for idx, e in df_r.iterrows():

        if e.dif==1:
            tiempos_arriba_der.append(idx)
        if e.dif==-1:
            tiempos_abajo_der.append(idx)
        
    tiempos_arriba_der = np.array(tiempos_arriba_der)
    tiempos_abajo_der = np.array(tiempos_abajo_der)

    min_len = min(len(tiempos_arriba_der), len(tiempos_abajo_der))
    derecho = tiempos_abajo_der[:min_len-1] - tiempos_arriba_der[:min_len-1]

    #ojo izquierdo
    tiempos_arriba_izq = []
    tiempos_abajo_izq = []
    for idx, e in df_l.iterrows():

        if e.dif==1:
            tiempos_arriba_izq.append(idx)
        if e.dif==-1:
            tiempos_abajo_izq.append(idx)
        
    tiempos_arriba_izq = np.array(tiempos_arriba_izq)
    tiempos_abajo_izq = np.array(tiempos_abajo_izq)

    min_len = min(len(tiempos_arriba_izq), len(tiempos_abajo_izq))
    izquierdo =  tiempos_abajo_izq[:min_len-1] - tiempos_arriba_izq[:min_len-1]

    #excluyo los bordes los bordes
    derecho = derecho[1:-1]
    izquierdo = izquierdo[1:-1]

    results = {
        'derecho':derecho,
        'izquierdo':izquierdo
    }

    return results

def V_prima(V, t,coefs):
    """
    coefs tiene que ir en este orden: [M, a , e , g, L, tau, tauh, R]
    """
    M = coefs[0]
    a = coefs[1]
    e = coefs[2]
    g = coefs[3]
    L = coefs[4]
    tau = coefs[5]
    tauh = coefs[6]
    R = coefs[7]
    El = V[0]
    Er = V[1]
    Hl = V[2]
    Hr = V[3]

    El_temp = (L -a*Er + e*El -g*Hl)
    El_temp = El_temp if El_temp>0 else 0

    Er_temp = (R - a*El + e*Er - g*Hr)
    Er_temp = Er_temp if Er_temp>0 else 0

    El_prima = (-El + M*El_temp)/tau
    Er_prima = (-Er + M*Er_temp)/tau
    Hl_prima = (-Hl + El)/tauh
    Hr_prima = (-Hr + Er)/tauh
    resultado = np.array([El_prima, Er_prima, Hl_prima, Hr_prima])
    
    return(resultado)

def V_prima_random(V, t,coefs):
    """
    Se incluye un factor estocastico normal, con media=0
    coefs tiene que ir en este orden: [M, a , e , g, L, tau, tauh, R, sigma]
    """
    M = coefs[0]
    a = coefs[1]
    e = coefs[2]
    g = coefs[3]
    L = coefs[4]
    tau = coefs[5]
    tauh = coefs[6]
    R = coefs[7]
    sigma = coefs[8]
    El = V[0]
    Er = V[1]
    Hl = V[2]
    Hr = V[3]

    El_temp = (L -a*Er + e*El -g*Hl)
    El_temp = El_temp if El_temp>0 else 0

    Er_temp = (R - a*El + e*Er - g*Hr)
    Er_temp = Er_temp if Er_temp>0 else 0

    random_kick = sigma * np.random.randn()

    El_prima = (-El + M*El_temp)/tau
    Er_prima = (-Er + M*Er_temp)/tau
    Hl_prima = (-Hl + El)/tauh + random_kick
    Hr_prima = (-Hr + Er)/tauh + random_kick
    resultado = np.array([El_prima, Er_prima, Hl_prima, Hr_prima])
    
    return(resultado)