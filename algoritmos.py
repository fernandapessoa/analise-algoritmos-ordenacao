import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from colorama import Fore, Style
import colorama

# Funções para gerar vetores conforme especificado
def generate_random_vector(n):
    return [random.randint(0, n**2) for _ in range(n)]

def generate_reverse_vector(n):
    return list(range(n, 0, -1))

def generate_sorted_vector(n):
    return list(range(1, n + 1))

def generate_nearly_sorted_vector(n):
    vector = list(range(1, n + 1))
    num_swaps = max(1, n // 10)
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        vector[i], vector[j] = vector[j], vector[i]
    return vector

# Algoritmos de ordenação
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

def counting_sort(arr):
    max_val = max(arr)
    min_val = min(arr)
    range_of_elements = max_val - min_val + 1
    count = [0] * range_of_elements
    output = [0] * len(arr)

    for i in range(len(arr)):
        count[arr[i] - min_val] += 1

    for i in range(1, len(count)):
        count[i] += count[i-1]

    for i in range(len(arr)-1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    for i in range(len(arr)):
        arr[i] = output[i]

    return arr

# Função para medir o tempo de execução de um algoritmo
def measure_time_and_sort(sort_func, arr):
    start_time = time.time()
    sorted_arr = sort_func(arr.copy())
    exec_time = time.time() - start_time
    return sorted_arr, exec_time

# Função para gerar e plotar os resultados
def generate_data_and_plot(inc, fim, stp, rpt, output_dir):
    algorithms = {
        "Bubble": bubble_sort,
        "Insertion": insertion_sort,
        "Merge": merge_sort,
        "Heap": heap_sort,
        "Quick": quick_sort,
        "Counting": counting_sort
    }

    categories = {
        "RANDOM": generate_random_vector,
        "REVERSE": generate_reverse_vector,
        "SORTED": generate_sorted_vector,
        "NEARLY_SORTED": generate_nearly_sorted_vector
    }

    # Criação do diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category_name, generate_vector in categories.items():
        print(f"\n\nIniciando categoria: {category_name} -------------------------  ")
        results = []

        for n in range(inc, fim + 1, stp):
            print(f"\n  Processando vetor de tamanho {n}...")
            avg_times = {'n': n}

            for name, sort_func in algorithms.items():
                print(f"\n    Executando {name} sort...")
                total_time = 0
                for i in range(rpt):
                    vector = generate_vector(n)
                    _, exec_time = measure_time_and_sort(sort_func, vector)
                    total_time += exec_time
                    print(f"      Repetição {i+1}/{rpt} - Tempo: {exec_time:.6f} s")

                avg_times[name] = total_time / rpt
                print(f"    Média de tempo para {name} sort: {avg_times[name]:.6f} s")

            results.append(avg_times)

        df = pd.DataFrame(results)

        # Gerando gráfico
        plt.figure(figsize=(10, 6))
        for column in df.columns[1:]:
            plt.plot(df['n'], df[column], label=column)

        plt.xlabel('Tamanho do Vetor (n)')
        plt.ylabel('Tempo de Execução (s)')
        plt.title(f'Comparação dos Algoritmos de Ordenação - {category_name}')
        plt.legend()
        plt.grid(True)

        # Salvando o gráfico como PNG na pasta de saída
        graph_filename = os.path.join(output_dir, f'comparison_chart_{category_name}.png')
        plt.savefig(graph_filename)
        print(f"Gráfico salvo como '{graph_filename}'")

        plt.show()

        # Salvando a tabela em formato CSV na pasta de saída
        csv_filename = os.path.join(output_dir, f'comparison_table_{category_name}.csv')
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

    generate_data_and_plot(args.inc, args.fim, args.stp, args.rpt, output_dir)

if __name__ == "__main__":
    main()
