# -----------------------------------------------------------------------------  
# DEMODULADOR MSK  
# -----------------------------------------------------------------------------  

import numpy as np
import scipy.signal as signal

# ------------------------------------------------------------------------------
# Filtros
# ------------------------------------------------------------------------------

def filtro_passa_baixa(freq_corte, fs, ordem=5):
    """Filtro Butterworth passa-baixa."""
    nyquist = 0.5 * fs
    normalizado = freq_corte / nyquist
    return signal.butter(ordem, normalizado, btype='low')

def filtro_passa_alta(freq_corte, fs, ordem=5):
    """Filtro Butterworth passa-alta."""
    nyquist = 0.5 * fs
    normalizado = freq_corte / nyquist
    return signal.butter(ordem, normalizado, btype='high')


# ------------------------------------------------------------------------------
# Portadoras I/Q
# ------------------------------------------------------------------------------

def gerar_portadora_MSK_base(Fs, Fc, Baud, total_samples, fase=0, Teste=0):
    """Gera portadoras I/Q para MSK com parâmetros opcionais de teste."""
    t = np.arange(total_samples) / Fs
    M = 4
    Tb = 1 / (Baud * np.log2(M))
    Fck = 1 / (4 * Tb)
    A = np.sqrt(1 / (2 * (1 / (2 * Baud)))) / 4
    
    if isinstance(fase, (float, int)):  # Se for escalar
       fase_MSK = fase
       fase_port = fase
    else:
       fase = np.asarray(fase)
       if len(fase) != len(t):
           raise ValueError("fase (GPS) deve ter o mesmo número de amostras que o sinal")
       fase_MSK = fase
       fase_port = fase

    argumento_MSK = 2 * np.pi * Fck * t
    argumento_portadora = 2 * np.pi * Fc * t

    msk_cos = np.cos(argumento_MSK + fase_MSK)
    msk_sin = np.sin(argumento_MSK + fase_MSK)
    port_cos = np.cos(argumento_portadora + fase_port)
    port_sin = np.sin(argumento_portadora + fase_port)

    if Teste == 1:
        msk_cos = np.abs(msk_cos)
        msk_sin = np.abs(msk_sin)
    elif Teste == 2:
        port_cos = np.abs(port_cos)
        port_sin = np.abs(port_sin)
    elif Teste == 3:
        msk_cos = np.abs(msk_cos)
        msk_sin = np.abs(msk_sin)
        port_cos = np.abs(port_cos)
        port_sin = np.abs(port_sin)

    sinal_I = msk_cos * port_cos / A
    sinal_Q = -msk_sin * port_sin / A

    return sinal_I, sinal_Q


# ------------------------------------------------------------------------------
# Integração por bit
# ------------------------------------------------------------------------------

def integrar_canal(sinal, N_bit, start=0):
    total = len(sinal)
    blocos = total // N_bit
    integrados = []
    for k in np.arange(start, blocos):
        ini = k * N_bit
        fim = ini + N_bit
        if fim <= total:
            bloco = sinal[ini:fim]
            integrados.append(np.sum(bloco) / N_bit)
    return np.array(integrados)


# ------------------------------------------------------------------------------
# Decisão de bit com base na fase
# ------------------------------------------------------------------------------

def decisor_de_fase(Li, Lq, impar=False):
    Th0 = 0 if Li > 0 else np.pi
    ThPI = -np.pi/2 if Lq > 0 else np.pi/2
    
    if not impar:
        Vec_bit = [Th0, ThPI]
    else:
        Vec_bit = [ThPI, Th0]

    if Vec_bit == [0, np.pi/2]: return 1, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [np.pi/2, 0]: return 0, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [np.pi/2, np.pi]: return 1, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [np.pi, np.pi/2]: return 0, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [0, -np.pi/2]: return 0, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [-np.pi/2, 0]: return 1, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [-np.pi/2, np.pi]: return 0, Vec_bit[0], Vec_bit[1]
    elif Vec_bit == [np.pi, -np.pi/2]: return 1, Vec_bit[0], Vec_bit[1]
    else: return -1

# ------------------------------------------------------------------------------
# Demodulação Principal
# ------------------------------------------------------------------------------

def demodular_MSK2(sinal_VLF, sinal_CGPS, Fs, Rs, Fc, GPS=False, extrair_ascii=False, Teste=0):
    """
    Demodulador MSK para sinais VLF.
    """

    Rb = Rs * 2
    Tb = 1 / Rb
    N_bit = int(Fs * Tb)
    total_samples = len(sinal_VLF)

    # Geração de portadoras
    fase_gps = sinal_CGPS if GPS else 0
    portadora_sin, portadora_cos = gerar_portadora_MSK_base(
        Fs, Fc, Rs, total_samples, fase=fase_gps, Teste=Teste
    )

    # Filtro passa-alta para remover esferics
    b_fase, a_fase = filtro_passa_alta(12000, Fs)
    sinal_filtrado = signal.filtfilt(b_fase, a_fase, sinal_VLF)

    # Modulação I/Q
    sinal_I = sinal_filtrado * portadora_sin
    sinal_Q = sinal_filtrado * portadora_cos

    # Filtro passa-baixa
    b_lp, a_lp = filtro_passa_baixa(Rs, Fs)
    I_filtrado = 2 * signal.filtfilt(b_lp, a_lp, sinal_I)
    Q_filtrado = 2 * signal.filtfilt(b_lp, a_lp, sinal_Q)

    # Integração por símbolo
    simbolos_I = integrar_canal(I_filtrado, N_bit, start=0)
    simbolos_Q = integrar_canal(Q_filtrado, N_bit, start=1)

    # Fase integrada (plano IQ)
    fase_integrada = np.angle(simbolos_I[:len(simbolos_Q)] + 1j * simbolos_Q[:len(simbolos_I)])

    # Decodificação dos bits
    bits_recuperados = []
    fase_esperada = []
    for k in range(min(len(simbolos_I) - 1, len(simbolos_Q))):
        if k % 2 == 0:
            bit, th0, thpi = decisor_de_fase(simbolos_I[k], simbolos_Q[k], impar=False)
        else:
            bit, thpi, th0 = decisor_de_fase(simbolos_I[k+1], simbolos_Q[k], impar=True)
        if bit != -1:
            bits_recuperados.append(bit)
            fase_esperada.append(th0 if k % 2 == 0 else thpi)

    # ASCII opcional
    ASCII72 = []
    if extrair_ascii:
        bytes_rec7 = [
            int("".join(str(b) for b in bits_recuperados[i:i+7]), 2)
            for i in range(0, len(bits_recuperados) - 7, 7)
        ]
        ASCII72 = [
            val for val in bytes_rec7
            if (32 <= val <= 96) or (123 <= val <= 126)
        ]

    # Amplitude vetorial
    Amp = np.sqrt(simbolos_Q[:len(simbolos_I)]**2 + simbolos_I[:len(simbolos_Q)]**2)

    return (
        np.array(bits_recuperados),
        np.array(ASCII72),
        np.array(fase_esperada),
        np.array(fase_integrada),
        np.array(Amp),
        np.array(simbolos_I),
        np.array(simbolos_Q)
    )
