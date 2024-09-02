import random
import os
import argparse

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

# Função para salvar o vetor em um arquivo txt
def save_vector_to_file(vector, filename):
    with open(filename, 'w') as f:
        for number in vector:
            f.write(f"{number}\n")

# Função principal para gerar todos os casos de teste
def generate_test_cases(inc, fim, stp, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vector_generators = [
        ("RANDOM", generate_random_vector),
        ("REVERSE", generate_reverse_vector),
        ("SORTED", generate_sorted_vector),
        ("NEARLY_SORTED", generate_nearly_sorted_vector)
    ]

    for label, generate_vector in vector_generators:
        for n in range(inc, fim + 1, stp):
            filename = os.path.join(output_dir, f"{label}_{n}.txt")
            vector = generate_vector(n)
            save_vector_to_file(vector, filename)
            print(f"Generated {filename}")

# Função para processar argumentos da linha de comando
def main():
    parser = argparse.ArgumentParser(description='Gerador de casos de teste para algoritmos de ordenação.')
    parser.add_argument('inc', type=int, help='Tamanho inicial do vetor.')
    parser.add_argument('fim', type=int, help='Tamanho final do vetor.')
    parser.add_argument('stp', type=int, help='Intervalo entre os tamanhos dos vetores.')
    parser.add_argument('--output_dir', type=str, default='test_cases', help='Diretório de saída para os arquivos gerados.')

    args = parser.parse_args()

    generate_test_cases(args.inc, args.fim, args.stp, args.output_dir)

if __name__ == "__main__":
    main()
