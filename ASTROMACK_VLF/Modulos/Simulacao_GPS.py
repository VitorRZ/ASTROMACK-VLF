# -*- coding: utf-8 -*-
"""
Created on Thu May 22 00:53:52 2025
@author: VITOR
"""
import numpy as np

# ------------------------------------------------------------------------------
# GERADOR DE PULSOS GPS COM JITTER CONTROLADO
# ------------------------------------------------------------------------------

def gerar_pulso_GPS(sample_rate, jitter_range_ms, tap):
    """
    Gera um sinal com pulsos de 1 segundo simulando um GPS com jitter.

    Parâmetros:
        sample_rate (int): taxa de amostragem em Hz (ex: 96000)
        jitter_range_ms (float): jitter em milissegundos
        tap (int): número total de amostras a serem geradas

    Retorno:
        pulso_GPS (np.array): array com pulsos de valor 1 nas posições sincronizadas
    """
    pulso_GPS = np.zeros(tap - sample_rate, dtype=np.float32)  # -1 segundo
    jitter_range_sec = jitter_range_ms / 1000.0

    secs = np.arange(tap - sample_rate)  # cada "segundo virtual"
    jitter = np.random.uniform(-jitter_range_sec, jitter_range_sec, size=secs.shape)
    pulse_times = secs + jitter
    pulse_indices = (pulse_times * sample_rate).astype(int)

    # Garante que os índices estão dentro do array
    pulse_indices = pulse_indices[(pulse_indices >= 0) & (pulse_indices < len(pulso_GPS))]

    pulso_GPS[pulse_indices] = 1.0
    return pulso_GPS
