# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:17:24 2020

@author: Camila
"""

import numpy as np
import pandas as pd

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

lista_dados = pd.read_csv("C:/Users/Dell/Desktop/basededados.csv")

base =  lista_dados
target_base = lista_dados["Classificador"].values 
target = np.array(target_base)

entradas = base
valoresSaida = target

saidas = np.empty([2000, 1], dtype=int)
for i in range(2000):
    saidas[i] = valoresSaida[i]

pesos0 = 2*np.random.random((7,5)) - 1
pesos1 = 2*np.random.random((5,1)) - 1

epocas = 1000
taxaAprendizagem = 0.3
momento = 1


for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)


percentual_acerto = (float(100) - mediaAbsoluta)
print("Porcentagem atual de acerto: " + str(percentual_acerto) + "%")

