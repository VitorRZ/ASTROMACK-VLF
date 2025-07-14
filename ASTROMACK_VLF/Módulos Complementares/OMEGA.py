import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def histograma_caracteres_legiveis(lista, nome, limite=None, top=30):
    dados = lista if limite is None else lista[:limite]
    legiveis = [chr(c) for c in dados if 32 <= c <= 126]
    contagem = Counter(legiveis)
    mais_comuns = contagem.most_common(top)
    key=[]
    freq = []

    #print(f"\nTop {top} caracteres mais frequentes ({nome}):")
    for char, count in mais_comuns:
        key.append(char)
        freq.append(count)
        #print(f"  '{char}': {count}")
        
    '''
    # Gráfico
    plt.figure(figsize=(12, 5))
    chars, freqs = zip(*mais_comuns)
    plt.bar(chars, freqs, color='skyblue')
    plt.title(f"Top {top} caracteres mais comuns - {nome}")
    plt.xlabel("Caractere")
    plt.ylabel("Frequência")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    '''
    return key, freq

# Função para converter Decimal em bits, utilizada para recuperar o bit de referencia
def gerar_bits_personalizado(lista_ascii):
    bits = np.concatenate([np.array(list(f"{n:07b}"), dtype=int) for n in lista_ascii])
    return bits


def single_pole_iir_filter(input_signal, tau, sample_rate):
    """
    Aplica um filtro IIR de um único pólo a um sinal de entrada.

    - input_signal: Sinal a ser filtrado (array).
    - tau: Tempo de integração (em segundos).
    - sample_rate: Taxa de amostragem do sinal (Hz).
    
    Retorna:
    - sinal filtrado.
    """
    # Calculando o parâmetro alpha
    alpha = np.exp(-1 / (tau * sample_rate))

    # Inicializando o sinal de saída
    output_signal = np.zeros_like(input_signal)

    # Aplicando o filtro
    for i in range(1, len(input_signal)):
        output_signal[i] = (1 - alpha) * input_signal[i] + alpha * output_signal[i - 1]

    return output_signal

def mapa_densidade(lista, padrao, bloco_tamanho=10000):
    texto = ''.join(chr(c) for c in lista if 32 <= c <= 126)
    #tamanho = len(padrao)
    blocos = len(texto) // bloco_tamanho
    densidades = []

    for i in range(blocos):
        trecho = texto[i * bloco_tamanho:(i + 1) * bloco_tamanho]
        contagem = trecho.count(padrao)
        densidades.append(contagem)
    '''
    t=np.arange(0,24,24/len(densidades))
    plt.figure(figsize=(12, 4))
    plt.plot(t,densidades)
    plt.title(f"Densidade do padrão '{padrao}'")
    plt.xlabel("horas")
    plt.ylabel("Ocorrências")
    plt.grid(True)
    plt.show()
    '''


def Eleitor_de_bit_piloto(ASCII2):
    

    key, freq =histograma_caracteres_legiveis(ASCII2, "7-bit (ASCII2)", top=150)

    # Verificando a Letra que se reperte com variação estavel e constante
    
    # Verifica o maximo
    i=0
    while max(freq)<= freq[i] and i<len(freq)-1:
            i+=1
    k=0
    while chr(k)<= key[i]:
            k+=1
    k-=1
    
    
    
    bit_REF_max = gerar_bits_personalizado([k])
    
    # Verifica o minimo
    
    i=0
    while min(freq)<= freq[i] and i<len(freq)-1:
            i+=1
    k=0
    while chr(k)<= key[i]:
            k+=1
    k-=1
    
    bit_REF = gerar_bits_personalizado([k])
    
    mapa_densidade(ASCII2, key[i])
    
    return bit_REF, bit_REF_max



def OMEGA(bit_REF, bitss):
    dt = 0
    delT = np.zeros_like(bitss)
    for i in np.arange(7,len(bitss),1):
        if bit_REF[0]==bitss[i-6] and bit_REF[1]==bitss[i-5] and bit_REF[1]==bitss[i-4] and bit_REF[1]==bitss[i-3] and bit_REF[1]==bitss[i-2] and bit_REF[1]==bitss[i-1] and bit_REF[1]==bitss[i-0]:
            dt = 0
        else:
            dt-=1
        delT[i] = dt
    
    #plt.figure(figsize=(12, 5))
    #plt.plot(delT)
    
    i=0
    kk=0
    suavizacao = 2
    sdelT = np.zeros(len(delT)//(400*60*suavizacao))
    for i in np.arange(400*60*suavizacao,len(delT),400*60*suavizacao):
        sdelT[kk] = np.mean(delT[(i-400*60*suavizacao):i])
        kk+=1
    
    fase2 = single_pole_iir_filter(sdelT, 1/(400*10),96000)
    return fase2