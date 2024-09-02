import os
import time
import pandas as pd
import matplotlib.pyplot as plt

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

# Função para gerar gráficos e tabelas separadas para RANDOM, REVERSE e SORTED
def generate_data_and_plot(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    algorithms = {
        "Bubble": bubble_sort,
        "Insertion": insertion_sort,
        "Merge": merge_sort,
        "Heap": heap_sort,
        "Quick": quick_sort,
        "Counting": counting_sort
    }

    categories = {
        "RANDOM": [],
        "REVERSE": [],
        "SORTED": []
    }

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filepath}")

            # Identifica a categoria do arquivo
            if "RANDOM" in filename.upper():
                category = "RANDOM"
            elif "REVERSE" in filename.upper():
                category = "REVERSE"
            elif "SORTED" in filename.upper() or "ORDENADO" in filename.upper():
                category = "SORTED"
            else:
                print(f"Warning: Filename {filename} does not match any known category. Skipping.")
                continue
            
            with open(filepath, 'r') as f:
                arr = [int(line.strip()) for line in f]
                n = len(arr)

                results = {'n': n}

                for name, sort_func in algorithms.items():
                    sorted_arr, exec_time = measure_time_and_sort(sort_func, arr)
                    results[name] = exec_time

                    # Salva o vetor ordenado em um arquivo
                    output_filename = f"{filename.split('.')[0]}_{name}_sorted.txt"
                    output_filepath = os.path.join(output_dir, output_filename)
                    with open(output_filepath, 'w') as out_f:
                        for num in sorted_arr:
                            out_f.write(f"{num}\n")

                    print(f"Algoritmo: {name}, Categoria: {category}, Tamanho do vetor: {n}, Tempo de execução: {exec_time:.6f} segundos")

                categories[category].append(results)

    for category, results in categories.items():
        if results:
            df = pd.DataFrame(results)

            # Gerando gráfico
            plt.figure(figsize=(10, 6))
            for column in df.columns[1:]:
                plt.plot(df['n'], df[column], label=column)
            
            plt.xlabel('Tamanho do Vetor (n)')
            plt.ylabel('Tempo de Execução (s)')
            plt.title(f'Comparação dos Algoritmos de Ordenação - {category}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'comparison_chart_{category}.png'))
            plt.show()

            # Salvando a tabela em formato CSV
            df.to_csv(os.path.join(output_dir, f'comparison_table_{category}.csv'), index=False)

# Exemplo de uso
input_dir = "test_cases"  # Diretório onde os arquivos de entrada estão localizados
output_dir = "resultados"  # Diretório onde os gráficos e tabelas serão salvos

generate_data_and_plot(input_dir, output_dir)
