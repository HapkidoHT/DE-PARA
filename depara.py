import pandas as pd
import re
import nltk
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Baixar recursos do NLTK
nltk.download('stopwords')

# Função para normalizar texto
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def advanced_normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Função para combinar múltiplas métricas de similaridade
def combined_similarity(text1, text2):
    # Similaridade de cosseno
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Similaridade Fuzzy
    fuzzy_score = fuzz.ratio(text1, text2) / 100
    
    # Similaridade média ponderada (ajustar pesos conforme necessário)
    combined_score = (cosine_sim * 0.7) + (fuzzy_score * 0.3)
    
    return combined_score

# Função para selecionar arquivo
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        file_label.config(text=file_path)
        process_button.config(state=tk.NORMAL)

# Função para processar o arquivo
def process_file():
    file_path = file_label.cget("text")
    sheet_antiga = sheet_antiga_entry.get()
    sheet_nova = sheet_nova_entry.get()
    coluna_antiga = coluna_antiga_entry.get()
    coluna_nova = coluna_nova_entry.get()
    
    # Obter valores dos sliders
    distance_threshold = distance_threshold_slider.get()
    combined_score_threshold = combined_score_threshold_slider.get()

    if file_path and sheet_antiga and sheet_nova and coluna_antiga and coluna_nova:
        # Carregar os dados das planilhas especificadas
        antiga_df = pd.read_excel(file_path, sheet_name=sheet_antiga)
        nova_df = pd.read_excel(file_path, sheet_name=sheet_nova)

        # Normalizar os modelos nas duas planilhas
        antiga_df['Modelo_Normalizado'] = antiga_df[coluna_antiga].apply(advanced_normalize)
        nova_df['Modelo_Normalizado'] = nova_df[coluna_nova].apply(advanced_normalize)

        # Criar o modelo TF-IDF para vetorização
        vectorizer = TfidfVectorizer()
        antiga_vectors = vectorizer.fit_transform(antiga_df['Modelo_Normalizado'])

        # Usar NearestNeighbors para encontrar os melhores matches
        nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
        nn.fit(antiga_vectors)

        # Vetorizar a planilha nova
        nova_vectors = vectorizer.transform(nova_df['Modelo_Normalizado'])

        # Encontrar as melhores correspondências
        distances, indices = nn.kneighbors(nova_vectors)

        # Adicionar uma nova coluna 'Mais_Compatível' na planilha 'Nova'
        nova_df['Mais_Compatível'] = ""

        # Configurar a barra de progresso
        progress_bar['maximum'] = len(nova_df)
        progress_bar['value'] = 0

        for i, (distance, index) in enumerate(zip(distances, indices)):
            # Ajuste de limite de distância de cosseno e combinação com Fuzzy
            if distance[0] < distance_threshold:  # Ajuste usando o slider
                modelo_antigo = antiga_df.iloc[index[0]][coluna_antiga]
                modelo_novo = nova_df.iloc[i][coluna_nova]
                combined_score = combined_similarity(modelo_antigo, modelo_novo)
                
                if combined_score > combined_score_threshold:  # Ajuste usando o slider
                    nova_df.at[i, 'Mais_Compatível'] = modelo_antigo
                else:
                    nova_df.at[i, 'Mais_Compatível'] = "No Match"
            else:
                nova_df.at[i, 'Mais_Compatível'] = "No Match"

            # Atualizar a barra de progresso
            progress_bar['value'] = i + 1
            root.update_idletasks()

        # Remover colunas normalizadas antes de salvar
        nova_df = nova_df.drop(columns=['Modelo_Normalizado'])

        # Salvar o dataframe atualizado em um novo arquivo Excel
        output_file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if output_file_path:
            nova_df.to_excel(output_file_path, index=False)
            messagebox.showinfo("Processamento Completo", f"Arquivo salvo em: {output_file_path}")
    else:
        messagebox.showerror("Erro", "Por favor, preencha todos os campos.")

# Função para limpar todos os campos
def clear_all():
    file_label.config(text="Nenhum arquivo selecionado")
    sheet_antiga_entry.delete(0, tk.END)
    sheet_nova_entry.delete(0, tk.END)
    coluna_antiga_entry.delete(0, tk.END)
    coluna_nova_entry.delete(0, tk.END)
    progress_bar['value'] = 0
    process_button.config(state=tk.DISABLED)

# Configurar a interface gráfica
root = tk.Tk()
root.title("Douglas - Comparação de Modelos ML")
root.geometry('600x700')
root.resizable(False, False)

style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12), padding=6)
style.configure('TEntry', font=('Helvetica', 12))
style.configure('TProgressbar', thickness=20)
style.configure('TFrame', background='#f0f0f0')
style.configure('TButton', background='#4CAF50', foreground='black', focuscolor='none')

frame = ttk.Frame(root, padding="20 20 20 20")
frame.pack(expand=True, fill='both')

title_label = ttk.Label(frame, text="Comparação de Modelos de Planilhas", font=('Helvetica', 16, 'bold'))
title_label.pack(pady=10)

file_label = ttk.Label(frame, text="Nenhum arquivo selecionado", wraplength=400, background='#f0f0f0')
file_label.pack(pady=10)

select_button = ttk.Button(frame, text="Selecionar Arquivo", command=select_file)
select_button.pack(pady=5)

ttk.Label(frame, text="Nome da planilha (Sheet1):", background='#f0f0f0').pack(pady=5)
sheet_antiga_entry = ttk.Entry(frame)
sheet_antiga_entry.pack(pady=5)

ttk.Label(frame, text="Nome da Coluna 1:", background='#f0f0f0').pack(pady=5)
coluna_antiga_entry = ttk.Entry(frame)
coluna_antiga_entry.pack(pady=5)

ttk.Label(frame, text="Nome da planilha (Sheet2):", background='#f0f0f0').pack(pady=5)
sheet_nova_entry = ttk.Entry(frame)
sheet_nova_entry.pack(pady=5)

ttk.Label(frame, text="Nome da Coluna 2:", background='#f0f0f0').pack(pady=5)
coluna_nova_entry = ttk.Entry(frame)
coluna_nova_entry.pack(pady=5)

# Adicionar controles deslizantes para ajustar limites de distância e pontuação combinada
distance_threshold_label = ttk.Label(frame, text="Limite de Distância de Cosseno (0 a 1):", background='#f0f0f0')
distance_threshold_label.pack(pady=5)
distance_threshold_slider = ttk.Scale(frame, from_=0, to=1, orient='horizontal', length=300)
distance_threshold_slider.set(0.5)  # Valor inicial
distance_threshold_slider.pack(pady=5)

combined_score_threshold_label = ttk.Label(frame, text="Limite de Pontuação Combinada (0 a 1):", background='#f0f0f0')
combined_score_threshold_label.pack(pady=5)
combined_score_threshold_slider = ttk.Scale(frame, from_=0, to=1, orient='horizontal', length=300)
combined_score_threshold_slider.set(0.5)  # Valor inicial
combined_score_threshold_slider.pack(pady=5)

process_button = ttk.Button(frame, text="Processar Arquivo", state=tk.DISABLED, command=process_file)
process_button.pack(pady=5)

clear_button = ttk.Button(frame, text="Limpar Tudo", command=clear_all)
clear_button.pack(pady=5)

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()
