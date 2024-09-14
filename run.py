import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from colorama import Fore, Style
import colorama

# Gerar vetores ------------------------------------------------------------------------------------------------------------

#Sorted List
#obs: executar apenas 1x
def gerar_vetor_ordenado(n): 
    #de 1 a n
    return list(range(1,n+1)) 

#obs: deve ser executado apenas 1 vez
def gerar_vetor_reverso(n): 
    #varia de n até 1
    return list(range(n, 0, -1)) 

def gerar_vetor_aleatorio(n):
    #n numeros variando de 0 até n**2, que nem o pdf pediu
    return [random.randint(0, n) for _ in range(n)] 


def gerar_vetor_quase_ordenado(n):
    # Gerar n números inteiros pseudoaleatórios no intervalo [0, n²]
    vetor = sorted(random.randint(0, n) for _ in range(n))
    
    # Definir 10% dos elementos para serem embaralhados
    num_trocas = int(0.1 * n) 
    
    for _ in range(num_trocas):
        i, j = random.sample(range(n), 2)
        vetor[i], vetor[j] = vetor[j], vetor[i]
    
    return vetor



#Algoritmos --------------------------------------------------------------------------------------------------------------

# Bubble Sort ----------------------------------
def bubble_sort(numbers):
    size = len(numbers)
    for i in range(size -1): 
        for j in range (size -1):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]

# Insertion Sort ----------------------------------
def insertion_sort(vetor):
    for j in range(1, len(vetor)):
        key = vetor[j]
        i = j - 1
        while i >= 0 and vetor[i] > key:
            vetor[i + 1] = vetor[i]
            i -= 1
        vetor[i + 1] = key


# Merge Sort ----------------------------------
def merge(A, L, R):
    i = 0
    j = 0
    k = 0

    # Enquanto houver elementos em L e R
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1

    while i < len(L):
        A[k] = L[i]
        i += 1
        k += 1

    while j < len(R):
        A[k] = R[j]
        j += 1
        k += 1

def merge_sort(A):
    if len(A) > 1:
        mid = len(A) // 2  # Ponto médio do vetor
        L = A[:mid]  # Metade esquerda
        R = A[mid:]  # Metade direita

        merge_sort(L)
        merge_sort(R)

        merge(A, L, R)


# Heap Sort --------------------------------------------
def parent(i):
    return (i // 2)

def left(i):
    return 2 * i

def right(i):
    return 2 * i + 1

def max_heapify(A, i, heap_size):
    l = left(i)
    r = right(i)
    largest = i
    if l <= heap_size and A[l - 1] > A[i - 1]:
        largest = l
    if r <= heap_size and A[r - 1] > A[largest - 1]:
        largest = r
    if largest != i:
        A[i - 1], A[largest - 1] = A[largest - 1], A[i - 1]
        max_heapify(A, largest, heap_size)

def build_max_heap(A):
    heap_size = len(A)
    for i in range(len(A) // 2, 0, -1):
        max_heapify(A, i, heap_size)

def heap_sort(A):
    build_max_heap(A)
    heap_size = len(A)
    for i in range(len(A), 1, -1):
        A[0], A[i - 1] = A[i - 1], A[0]  # Troca o primeiro com o último
        heap_size -= 1
        max_heapify(A, 1, heap_size)


# Quick Sort ------------------------------------------------

def partition(A, p, r):
    pivot_index = random.randint(p, r)  # Escolhe um pivô aleatório
    A[pivot_index], A[r] = A[r], A[pivot_index]  # Troca o pivô com o último elemento
    x = A[r]  # Define o pivô como o último elemento (que agora contém o pivô aleatório)
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1

def quick_sort(A, p=0, r=None):
    if r is None:  # Verifica se r é None (chamada inicial)
        r = len(A) - 1  # Define r como o último índice do vetor
    if p < r:
        q = partition(A, p, r)  # Particiona e encontra a posição do pivô
        quick_sort(A, p, q - 1)  # Ordena a parte à esquerda do pivô
        quick_sort(A, q + 1, r)  # Ordena a parte à direita do pivô
        
        
# Counting Sort ------------------------------------------------
def counting_sort(A):
    k = max(A)
    B = [0] * len(A)
    C = [0] * (k + 1)

    for i in range(k):
        C[i] = 0

    for j in range(1, len(A)):
        C[A[j]] += 1

    for i in range(1, k + 1):
        C[i] += C[i - 1]

    # Passo 4: Construir o array B ordenado
    for j in range(len(A) - 1, -1, -1):
        B[C[A[j]] - 1] = A[j]
        C[A[j]] -= 1

    # Copiar o array B para A, agora ordenado
    for i in range(len(A)):
        A[i] = B[i]


# Funções auxiliazes --------------------------------------------------------------------------------------------------------------

def medir_tempo_e_executar(algoritmo_de_ordenacao, vetor):
    tempo_inicio = time.time()
    vetor_ordenado = algoritmo_de_ordenacao(vetor.copy())
    tempo_execucao = time.time() - tempo_inicio
    return tempo_execucao

# Chamadas dos algoritmos, medição de tempo e plotação de gráficos -----------------------------------------------------------------------------------

def processar_dados(inc, fim, stp, rpt, output_dir):
    algoritmos = {
        "Bubble": bubble_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort,
        "Heap Sort": heap_sort,
        "Quick Sort": quick_sort,
        "Counting Sort": counting_sort,
    }
    
    categorias_vetor = {
        "RANDOM": {"func": gerar_vetor_aleatorio, "repeat": rpt},
        "REVERSE": {"func": gerar_vetor_reverso, "repeat": 1},  # Executa apenas 1 vez
        "SORTED": {"func": gerar_vetor_ordenado, "repeat": 1},  # Executa apenas 1 vez
        "NEARLY_SORTED": {"func": gerar_vetor_quase_ordenado, "repeat": rpt}
    }

    # Criação do diretório de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Percorrendo os tipos de vetores
    for nome_categoria, info_categoria in categorias_vetor.items():
        print(f"\n\nIniciando categoria: {nome_categoria} -------------------------")
        resultados = []

        # Definindo o tamanho do vetor a ser gerado, começando com inc, terminando em fim e aumentando de stp em stp
        for n in range(inc, fim + 1, stp):
            print(f"\n  Processando vetor de tamanho {n}...")
            avg_times = {'n': n}

            # Para cada tamanho definido, vamos percorrer a lista de algoritmos
            for nome_algoritmo, algoritmo_ordenacao in algoritmos.items():
                print(f"\n    Executando {nome_algoritmo} sort...")
                total_time = 0

                # Para cada algoritmo de ordenação, vamos verar o vetor e calcular o tempo de execução
                # Os vetores sorted e reverse precisam ser executados apenas 1x, os demais de acordo com a variável rpt
                # O tempo de execução é armazenado na variável tempo total para posteriormente tirar a média 
                for i in range(info_categoria['repeat']):
                    vetor = info_categoria['func'](n)
                    exec_time = medir_tempo_e_executar(algoritmo_ordenacao, vetor)
                    total_time += exec_time
                    print(f"      Repetição {i+1}/{info_categoria['repeat']} - Tempo: {exec_time:.6f} s")

                # Ao final das repetições definidas, tiramos a média do tempo de execução, que é salva na lista de acordo com o nome do algoritmo
                avg_times[nome_algoritmo] = total_time / info_categoria['repeat']
                #print(f"    Média de tempo para {nome_algoritmo}: {avg_times[nome_algoritmo]:.6f} s")

            # Então, a lista de média do tempo de execução de cada algoritmo para cada tamanho de vetor é armazenada na lista resultados, formando uma matriz
            resultados.append(avg_times)

        print(resultados)

        # Criar DataFrame com os resultados
        df = pd.DataFrame(resultados)

        # Gerando gráfico
        plt.figure(figsize=(10, 6))
        for column in df.columns[1:]:
            plt.plot(df['n'], df[column], label=column)

        plt.xlabel('Tamanho do Vetor (n)')
        plt.ylabel('Tempo de Execução (s)')
        plt.title(f'Comparação dos Algoritmos de Ordenação - {nome_categoria}')
        plt.legend()
        plt.grid(True)

        # Salvando o gráfico como PNG
        graph_filename = os.path.join(output_dir, f'comparison_chart_{nome_categoria}.png')
        plt.savefig(graph_filename)
        print(f"Gráfico salvo como '{graph_filename}'")

        # Salvando a tabela como CSV
        csv_filename = os.path.join(output_dir, f'comparison_table_{nome_categoria}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Tabela de comparação salva como '{csv_filename}'")

# Função principal para processar argumentos da linha de comando
def main():
    parser = argparse.ArgumentParser(description='Análise de algoritmos de ordenação.')
    parser.add_argument('inc', type=int, help='Tamanho inicial do vetor.')
    parser.add_argument('fim', type=int, help='Tamanho final do vetor.')
    parser.add_argument('stp', type=int, help='Intervalo entre os tamanhos dos vetores.')
    parser.add_argument('rpt', type=int, help='Número de repetições para cada tamanho de vetor.')

    args = parser.parse_args()

    output_dir = "results"

    processar_dados(args.inc, args.fim, args.stp, args.rpt, output_dir)

if __name__ == "__main__":
    main()