import numpy as np
import scipy.io.wavfile as wav

morse_dict = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D",
    ".": "E", "..-.": "F", "--.": "G", "....": "H",
    "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P",
    "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z"
}

def simple_fft(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            result[k] += signal[n] * np.exp(angle)
    return np.abs(result)

def audio_to_spectra_matrix(file_path, block_duration=0.05):

    sampling_rate, signal = wav.read(file_path)

    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    block_size = int(block_duration * sampling_rate)
    num_blocks = len(signal) // block_size

    signal = signal[:num_blocks * block_size]

    blocks = signal.reshape((num_blocks, block_size))

    spectra = np.array([simple_fft(block) for block in blocks])

    return spectra

def make_sparse(matrix, threshold_ratio=0.1):
    threshold = np.max(matrix) * threshold_ratio
    return np.where(matrix > threshold, matrix, 0)

def matmul(A, B):
    m, n = A.shape
    print(B.shape)
    n_b, p = B.shape
    result = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def outer_product(u, v):
    m = u.shape[0]
    n = v.shape[0]
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            result[i, j] = u[i] * v[j]
    return result

def power_iteration(A, num_iterations=1000):
    rows, cols = A.shape
    b_k = np.random.rand(cols)
    b_k /= np.linalg.norm(b_k)

    for _ in range(num_iterations):
        # b_k1 = A @ b_k
        b_k1 = matmul(A, b_k.reshape(-1, 1)).flatten()
        # b_k1 = matmul(A, b_k)
        b_k1 = b_k1 / np.linalg.norm(b_k1)
        if np.linalg.norm(b_k1 - b_k) < 1e-10:
            break
        b_k = b_k1

    b_k_column = b_k.reshape(-1, 1)
    b_k_row = b_k.reshape(1, -1)

    numerator = matmul(b_k_row, matmul(A, b_k_column))[0][0]
    denominator = matmul(b_k_row, b_k_column)[0][0]
    ev = numerator / denominator
    return ev, b_k

def svd_manual(A, k=None):
    rows, cols = A.shape
    A_cp = A.copy().astype(float)
    r = min(rows, cols) if k is None else k

    U = np.zeros((rows, rows))
    S = np.zeros(r)
    V = np.zeros((cols, cols))

    for i in range(r):
        EV, v = power_iteration(matmul(A_cp.T, A_cp))
        v /= np.linalg.norm(v)
        sigma_i = np.sqrt(EV)
        S[i] = sigma_i
        u = matmul(A_cp, v.reshape(-1, 1)) / sigma_i
        u /= np.linalg.norm(u)
        U[:, i] = u
        V[:, i] = v
        A_cp -= sigma_i * outer_product(u, v)

    VT = V.T
    D = np.zeros((U.shape[1], VT.shape[0]))
    np.fill_diagonal(D, S)

    return U, D, VT

def reconstruct_matrix(U, Sigma_truncated, Vt):
    return matmul(U, matmul(Sigma_truncated, Vt))

def matrix_to_binary_signal(matrix):
    binary_signal = []

    a = sum(np.linalg.norm(row) for row in matrix)
    c = len(matrix)

    threshold = a/c

    for row in matrix:
        norm = np.linalg.norm(row)
        a+=norm
        c+=1
        if norm > threshold:
            binary_signal.append(1)
        else:
            binary_signal.append(0)

    return binary_signal

def binary_to_morse(binary_signal, dot_max=2, dash_min=3, letter_pause=3, word_pause=21):
    symbols = []
    i = 0
    while i < len(binary_signal):
        if binary_signal[i] == 1:
            length = 0
            while i < len(binary_signal) and binary_signal[i] == 1:
                length += 1
                i += 1
            if length <= dot_max:
                symbols.append(".")
            elif length >= dash_min:
                symbols.append("-")
        else:
            pause = 0
            while i < len(binary_signal) and binary_signal[i] == 0:
                pause += 1
                i += 1
            if pause >= word_pause:
                symbols.append(" / ")
            elif pause >= letter_pause:
                symbols.append(" ")

    return "".join(symbols)

def decode_morse(morse_code):
    words = morse_code.split(" / ")
    decoded_words = []

    for word in words:
        letters = word.strip().split(" ")
        decoded_word = ""
        for letter in letters:
            if letter:
                decoded_word += morse_dict.get(letter, "?")
        decoded_words.append(decoded_word)
    return " ".join(decoded_words)

file = 'helloworld.wav'

spectra = audio_to_spectra_matrix(file)

spectra_sparse = make_sparse(spectra)

U, Sigma, Vt = svd_manual(spectra_sparse, 20)

filtered_matrix = reconstruct_matrix(U, Sigma, Vt)

binary_signal = matrix_to_binary_signal(filtered_matrix)

morse_code = binary_to_morse(binary_signal)

print(decode_morse(morse_code))
