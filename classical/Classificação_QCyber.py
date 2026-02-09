# %%
"""
# Base Edge-IIoTset Cyber Security Dataset of IoT & IIoT
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Salva automaticamente todas as figuras em classical/figures quando plt.show() for chamado.
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
_FIGURE_SAVE_COUNT = 0


def _save_figures_instead_of_show(*args, **kwargs):
    global _FIGURE_SAVE_COUNT
    fig_nums = plt.get_fignums()
    for fig_num in fig_nums:
        fig = plt.figure(fig_num)
        _FIGURE_SAVE_COUNT += 1
        out_name = f"classificacao_qcyber_{_FIGURE_SAVE_COUNT:04d}.png"
        out_path = os.path.join(FIGURES_DIR, out_name)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[FIGURE] Saved: {out_path}")
    plt.close("all")


plt.show = _save_figures_instead_of_show

# Trio-only dataset policy: usa diretamente o CSV final do trio (source -> trio merged).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TRIO_DATASET_PATH = os.path.abspath(
    os.environ.get(
        "EDGE_TRIO_DATASET",
        os.path.join(_PROJECT_ROOT, "data", "processed", "edge_iot_trio_binary.csv"),
    )
)
_LEGACY_DATASET_FILENAMES = {
    "ML-EdgeIIoT-dataset.csv",
    "edge_pls8_debug.csv",
    "ML-EdgeIIoT-dataset-pp.csv",
    "CICIDS2017_Combinado.csv",
    "PCA_CICIDS2017.csv",
    "UNSW_NB15_Combinado.csv",
    "UNSW_NB15_Combinado_preprocessing.csv",
    "CIC-DDoS2019_Combinado.csv",
    "PCA_CIC-DDoS2019.csv",
    "unified_cicids_unsw_common.csv",
    "edge_pls8_full.csv",
}


def _validate_trio_dataset():
    if not os.path.exists(TRIO_DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset do trio não encontrado: {TRIO_DATASET_PATH}. "
            "Gere com scripts/prepare_trio_merged_dataset.py ou defina EDGE_TRIO_DATASET."
        )
_validate_trio_dataset()
_ORIGINAL_READ_CSV = pd.read_csv


def _read_csv_with_trio_redirection(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        basename = os.path.basename(filepath_or_buffer)
        if basename in _LEGACY_DATASET_FILENAMES:
            print(f"[TRIO] Redirecting '{basename}' -> '{TRIO_DATASET_PATH}'")
            filepath_or_buffer = TRIO_DATASET_PATH
    return _ORIGINAL_READ_CSV(filepath_or_buffer, *args, **kwargs)


pd.read_csv = _read_csv_with_trio_redirection

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('ML-EdgeIIoT-dataset.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'ML-EdgeIIoT-dataset.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
# MODIFICAÇÃO: Removendo tanto o alvo (Attack_type) quanto a coluna 'attack_label' das features
X = df.drop(columns=['Attack_type', 'Attack_label'])
y_raw = df['Attack_type']

# Checa e remove a coluna 'frame.time' se ela existir
if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

# Codifica colunas categóricas restantes em X
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codifica a variável alvo y
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 20% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados de 20%)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}              {result['cv_accuracy_std']:.4f}              {result['cv_f1_mean']:.4f}              {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

X_full_scaled = scaler.transform(X)
X_full_unscaled = X.values
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    # Dicionários para armazenar os scores de cada fold
    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        # Armazena scores gerais do fold
        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        # Armazena scores por classe do fold
        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    # --- Exibição das Métricas Gerais ---
    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    # --- Exibição das Métricas por Classe ---
    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<18} | {'Recall':<18} | {'F1-Score':<18}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=-1)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=-1)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)


# %%
"""
# Base Pre processada Edge-IIoTset Cyber Security Dataset of IoT & IIoT
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('edge_pls8_debug.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'ML-EdgeIIoT-dataset.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
# MODIFICAÇÃO: Removendo tanto o alvo (Attack_type) quanto a coluna 'attack_label' das features
X = df.drop(columns=['Attack_label', 'split'])
y_raw = df['Attack_label']

# Checa e remove a coluna 'frame.time' se ela existir
if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

# Codifica colunas categóricas restantes em X
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codifica a variável alvo y
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 20% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.5, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados de 20%)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}              {result['cv_accuracy_std']:.4f}              {result['cv_f1_mean']:.4f}              {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

X_full_scaled = scaler.transform(X)
X_full_unscaled = X.values
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    # Dicionários para armazenar os scores de cada fold
    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        # Armazena scores gerais do fold
        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        # Armazena scores por classe do fold
        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    # --- Exibição das Métricas Gerais ---
    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    # --- Exibição das Métricas por Classe ---
    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<18} | {'Recall':<18} | {'F1-Score':<18}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=-1)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=-1)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)


# %%
"""
# Base de dados IoTSec - OPF
"""

# %%
# pip install opfython

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# OPF
try:
    from opfython.models import SupervisedOPF
    from opfython.stream import loader
except ImportError:
    SupervisedOPF = None
    loader = None

from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---


class OPFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, distance='euclidean'):
        self.distance = distance
        self.model = None

    def fit(self, X, y):
        if SupervisedOPF is None:
            raise ImportError(
                "opfython não está disponível neste ambiente. "
                "Instale opfython em uma versão de Python compatível (<3.10)."
            )
        # Garante que X e y são arrays numpy
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        self.model = SupervisedOPF(distance=self.distance)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {'distance': self.distance}

    def set_params(self, **params):
        self.distance = params.get('distance', self.distance)
        return self

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('ML-EdgeIIoT-dataset-pp.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'ML-EdgeIIoT-dataset.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
# MODIFICAÇÃO: Removendo tanto o alvo (Attack_type) quanto a coluna 'attack_label' das features
X = df.drop(columns=['Attack_label'])
y_raw = df['Attack_label']

# Checa e remove a coluna 'frame.time' se ela existir
if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

# Codifica colunas categóricas restantes em X
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codifica a variável alvo y
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 20% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
     'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'use_scaled_data': False
    },

    #'SVM': {
    #    'model': SVC(random_state=42, probability=True),
    #    'params': {
    #        'C': [0.1, 1, 10, 100],
    #        'kernel': ['linear', 'rbf', 'poly'],
    #        'gamma': ['scale', 'auto']
    #    },
    #    'use_scaled_data': True
    #},

    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        },
        'use_scaled_data': True
    },

    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'use_scaled_data': True
    },

    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        'use_scaled_data': True
    },

    'OPF': {
          # Define com valor inicial válido
          'model': OPFClassifier(distance='euclidean'),
          'params': {
              'distance': ['additive_symmetric', 'average_euclidean', 'bhattacharyya', 'bray_curtis',
            'canberra', 'chebyshev', 'chi_squared', 'chord', 'clark', 'cosine',
            'dice', 'divergence', 'euclidean', 'gower', 'hamming', 'hassanat',
            'hellinger', 'jaccard', 'jeffreys', 'jensen', 'jensen_shannon',
            'k_divergence', 'kulczynski', 'kullback_leibler', 'lorentzian',
            'manhattan', 'matusita', 'mean_censored_euclidean', 'min_symmetric',
            'non_intersection', 'pearson', 'sangvi', 'soergel', 'squared_chord',
            'squared_euclidean', 'statistic', 'topsoe', 'vicis_symmetric1',
            'vicis_symmetric2', 'vicis_symmetric3', 'vicis_wave_hedges'
              ]
          },
          'use_scaled_data': True
      }
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados de 20%)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}              {result['cv_accuracy_std']:.4f}              {result['cv_f1_mean']:.4f}              {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

X_full_scaled = scaler.transform(X)
X_full_unscaled = X.values
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    # Dicionários para armazenar os scores de cada fold
    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        # Armazena scores gerais do fold
        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        # Armazena scores por classe do fold
        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    # --- Exibição das Métricas Gerais ---
    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    # --- Exibição das Métricas por Classe ---
    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<18} | {'Recall':<18} | {'F1-Score':<18}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=-1)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=-1)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)
print(cm_best)
#analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
# Base CIC-IDS-2017

"""

# %%
import pandas as pd
import glob

# --- Configuração ---
# O caminho para a pasta que contém os seus arquivos CSV.
# O padrão '/**/' faz uma busca recursiva em todas as subpastas.
pasta_dos_dados = 'Dataset'

# O nome do arquivo final que será gerado.
arquivo_de_saida = 'CICIDS2017_Combinado.csv'
# --------------------

print(f"Buscando arquivos .csv na pasta '{pasta_dos_dados}'...")

# 1. Encontra todos os caminhos de arquivos .csv de forma recursiva.
caminho_padrao = f"{pasta_dos_dados}/**/*.csv"
lista_de_arquivos = glob.glob(caminho_padrao, recursive=True)

if not lista_de_arquivos:
    print("⚠️ Nenhum arquivo .csv foi encontrado. Verifique o caminho da pasta.")
else:
    print(f"✅ Encontrados {len(lista_de_arquivos)} arquivos.")

    # 2. VERIFICAÇÃO DE COLUNAS
    print("\nVerificando se todos os arquivos têm as mesmas colunas...")
    try:
        # Pega as colunas do primeiro arquivo como referência, já limpando os espaços
        colunas_referencia = set(pd.read_csv(lista_de_arquivos[0], nrows=0).columns.str.strip())
        print(f"Colunas de referência encontradas e limpas no primeiro arquivo.")

        arquivos_validos = []
        for arquivo in lista_de_arquivos:
            # Lê apenas o cabeçalho (nrows=0) para ser rápido e limpa os espaços
            colunas_atuais = set(pd.read_csv(arquivo, nrows=0, on_bad_lines='skip').columns.str.strip())
            if colunas_atuais == colunas_referencia:
                arquivos_validos.append(arquivo)
            else:
                print(f"  ⚠️ Alerta: O arquivo '{arquivo}' tem colunas diferentes e será ignorado.")

    except Exception as e:
        print(f"❌ Erro crítico ao ler colunas de referência: {e}")
        arquivos_validos = [] # Zera a lista para não continuar

    # 3. COMBINA APENAS OS ARQUIVOS VÁLIDOS
    if not arquivos_validos:
        print("\n❌ Nenhum arquivo válido para combinar. O processo foi interrompido.")
    else:
        if len(arquivos_validos) != len(lista_de_arquivos):
             print(f"\n🔄 Combinando apenas os {len(arquivos_validos)} arquivos com colunas correspondentes...")
        else:
             print("\n✅ Todos os arquivos têm as mesmas colunas. Combinando...")

        # Usa uma "generator expression" para ler os arquivos sob demanda.
        dataframes_iterador = (pd.read_csv(f, on_bad_lines='warn') for f in arquivos_validos)

        # Concatena todos os DataFrames do iterador em um único DataFrame.
        df_combinado = pd.concat(dataframes_iterador, ignore_index=True)

        # Limpa espaços em branco de todos os nomes das colunas (causa comum de erros)
        df_combinado.columns = df_combinado.columns.str.strip()

        # 4. Salva o resultado em um novo arquivo CSV.
        df_combinado.to_csv(arquivo_de_saida, index=False)

        print("\n🎉 Processo concluído com sucesso!")
        print(f"💾 Arquivo final salvo como: '{arquivo_de_saida}'")
        print(f"📊 O arquivo combinado tem {df_combinado.shape[0]:,} linhas e {df_combinado.shape[1]} colunas.")



# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('CICIDS2017_Combinado.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'CICIDS2017_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['Label'])
y_raw = df['Label']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 10% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.10, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
# CICIDS 2017 Binário
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('PCA_CICIDS2017.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['Label'])
y_raw = df['Label']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%


# %%
"""
# Dataset UNSW_NB15
"""

# %%
import pandas as pd
import glob

# --- Configuração ---
# O caminho para a pasta que contém os seus arquivos CSV.
# O padrão '/**/' faz uma busca recursiva em todas as subpastas.
pasta_dos_dados = 'UNSW_NB15'

# O nome do arquivo final que será gerado.
arquivo_de_saida = 'UNSW_NB15_Combinado.csv'
# --------------------

print(f"Buscando arquivos .csv na pasta '{pasta_dos_dados}'...")

# 1. Encontra todos os caminhos de arquivos .csv de forma recursiva.
caminho_padrao = f"{pasta_dos_dados}/**/*.csv"
lista_de_arquivos = glob.glob(caminho_padrao, recursive=True)

if not lista_de_arquivos:
    print("⚠️ Nenhum arquivo .csv foi encontrado. Verifique o caminho da pasta.")
else:
    print(f"✅ Encontrados {len(lista_de_arquivos)} arquivos.")

    # 2. VERIFICAÇÃO DE COLUNAS
    print("\nVerificando se todos os arquivos têm as mesmas colunas...")
    try:
        # Pega as colunas do primeiro arquivo como referência, já limpando os espaços
        colunas_referencia = set(pd.read_csv(lista_de_arquivos[0], nrows=0).columns.str.strip())
        print(f"Colunas de referência encontradas e limpas no primeiro arquivo.")

        arquivos_validos = []
        for arquivo in lista_de_arquivos:
            # Lê apenas o cabeçalho (nrows=0) para ser rápido e limpa os espaços
            colunas_atuais = set(pd.read_csv(arquivo, nrows=0, on_bad_lines='skip').columns.str.strip())
            if colunas_atuais == colunas_referencia:
                arquivos_validos.append(arquivo)
            else:
                print(f"  ⚠️ Alerta: O arquivo '{arquivo}' tem colunas diferentes e será ignorado.")

    except Exception as e:
        print(f"❌ Erro crítico ao ler colunas de referência: {e}")
        arquivos_validos = [] # Zera a lista para não continuar

    # 3. COMBINA APENAS OS ARQUIVOS VÁLIDOS
    if not arquivos_validos:
        print("\n❌ Nenhum arquivo válido para combinar. O processo foi interrompido.")
    else:
        if len(arquivos_validos) != len(lista_de_arquivos):
             print(f"\n🔄 Combinando apenas os {len(arquivos_validos)} arquivos com colunas correspondentes...")
        else:
             print("\n✅ Todos os arquivos têm as mesmas colunas. Combinando...")

        # Usa uma "generator expression" para ler os arquivos sob demanda.
        dataframes_iterador = (pd.read_csv(f, on_bad_lines='warn') for f in arquivos_validos)

        # Concatena todos os DataFrames do iterador em um único DataFrame.
        df_combinado = pd.concat(dataframes_iterador, ignore_index=True)

        # Limpa espaços em branco de todos os nomes das colunas (causa comum de erros)
        df_combinado.columns = df_combinado.columns.str.strip()

        # 4. Salva o resultado em um novo arquivo CSV.
        df_combinado.to_csv(arquivo_de_saida, index=False)

        print("\n🎉 Processo concluído com sucesso!")
        print(f"💾 Arquivo final salvo como: '{arquivo_de_saida}'")
        print(f"📊 O arquivo combinado tem {df_combinado.shape[0]:,} linhas e {df_combinado.shape[1]} colunas.")



# %%
df = pd.read_csv('UNSW_NB15_Combinado.csv')
df.head(-100)

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('UNSW_NB15_Combinado.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['label','attack_cat'])
y_raw = df['attack_cat']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.30, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
# Dataset UNSW_NB15
"""

# %%
import pandas as pd
import glob

# --- Configuração ---
# O caminho para a pasta que contém os seus arquivos CSV.
# O padrão '/**/' faz uma busca recursiva em todas as subpastas.
pasta_dos_dados = 'UNSW_NB15-PP'

# O nome do arquivo final que será gerado.
arquivo_de_saida = 'UNSW_NB15_Combinado_preprocessing.csv'
# --------------------

print(f"Buscando arquivos .csv na pasta '{pasta_dos_dados}'...")

# 1. Encontra todos os caminhos de arquivos .csv de forma recursiva.
caminho_padrao = f"{pasta_dos_dados}/**/*.csv"
lista_de_arquivos = glob.glob(caminho_padrao, recursive=True)

if not lista_de_arquivos:
    print("⚠️ Nenhum arquivo .csv foi encontrado. Verifique o caminho da pasta.")
else:
    print(f"✅ Encontrados {len(lista_de_arquivos)} arquivos.")

    # 2. VERIFICAÇÃO DE COLUNAS
    print("\nVerificando se todos os arquivos têm as mesmas colunas...")
    try:
        # Pega as colunas do primeiro arquivo como referência, já limpando os espaços
        colunas_referencia = set(pd.read_csv(lista_de_arquivos[0], nrows=0).columns.str.strip())
        print(f"Colunas de referência encontradas e limpas no primeiro arquivo.")

        arquivos_validos = []
        for arquivo in lista_de_arquivos:
            # Lê apenas o cabeçalho (nrows=0) para ser rápido e limpa os espaços
            colunas_atuais = set(pd.read_csv(arquivo, nrows=0, on_bad_lines='skip').columns.str.strip())
            if colunas_atuais == colunas_referencia:
                arquivos_validos.append(arquivo)
            else:
                print(f"  ⚠️ Alerta: O arquivo '{arquivo}' tem colunas diferentes e será ignorado.")

    except Exception as e:
        print(f"❌ Erro crítico ao ler colunas de referência: {e}")
        arquivos_validos = [] # Zera a lista para não continuar

    # 3. COMBINA APENAS OS ARQUIVOS VÁLIDOS
    if not arquivos_validos:
        print("\n❌ Nenhum arquivo válido para combinar. O processo foi interrompido.")
    else:
        if len(arquivos_validos) != len(lista_de_arquivos):
             print(f"\n🔄 Combinando apenas os {len(arquivos_validos)} arquivos com colunas correspondentes...")
        else:
             print("\n✅ Todos os arquivos têm as mesmas colunas. Combinando...")

        # Usa uma "generator expression" para ler os arquivos sob demanda.
        dataframes_iterador = (pd.read_csv(f, on_bad_lines='warn') for f in arquivos_validos)

        # Concatena todos os DataFrames do iterador em um único DataFrame.
        df_combinado = pd.concat(dataframes_iterador, ignore_index=True)

        # Limpa espaços em branco de todos os nomes das colunas (causa comum de erros)
        df_combinado.columns = df_combinado.columns.str.strip()

        # 4. Salva o resultado em um novo arquivo CSV.
        df_combinado.to_csv(arquivo_de_saida, index=False)

        print("\n🎉 Processo concluído com sucesso!")
        print(f"💾 Arquivo final salvo como: '{arquivo_de_saida}'")
        print(f"📊 O arquivo combinado tem {df_combinado.shape[0]:,} linhas e {df_combinado.shape[1]} colunas.")



# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('UNSW_NB15_Combinado_preprocessing.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['label'])
y_raw = df['label']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.30, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
#  Dataset CIC-DDoS2019
"""

# %%
import pandas as pd

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('CIC-DDoS2019_Combinado.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['Label'])
y_raw = df['Label']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
# CIC-DDoS2019 - Binária
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('PCA_CIC-DDoS2019.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['Label'])
y_raw = df['Label']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
"""
# Base de dados - Combinada
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.

CORREÇÕES APLICADAS:
- Corrigido 'MemoryError' ao reduzir o paralelismo (n_jobs) durante a validação
  cruzada no dataset completo.
- Otimizado o uso de memória convertendo os dados para float32.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('unified_cicids_unsw_common.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'UNSW_NB15_Combinado.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
X = df.drop(columns=['attack_category','is_attack'])
y_raw = df['is_attack']

if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 30% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 'use_scaled_data': False},
    #'SVM': {'model': SVC(random_state=42, probability=True), 'params': {'C': [1, 10], 'kernel': ['rbf']}, 'use_scaled_data': True},
    #'Logistic Regression': {'model': LogisticRegression(random_state=42, max_iter=500), 'params': {'C': [0.1, 1, 10], 'solver': ['saga']}, 'use_scaled_data': True},
    #'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance']}, 'use_scaled_data': True}
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}             {result['cv_accuracy_std']:.4f}             {result['cv_f1_mean']:.4f}             {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

# CORREÇÃO: Converte os dados para float32 para economizar memória
X_full_scaled = scaler.transform(X).astype(np.float32)
X_full_unscaled = X.values.astype(np.float32)
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    # Usando um loop explícito para ter mais controle sobre o processo
    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<22} | {'Recall':<22} | {'F1-Score':<22}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError. Use n_jobs=1 se o erro persistir.
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=2)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    # CORREÇÃO: Reduzido n_jobs para evitar MemoryError.
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=2)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)

analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
df.head(100)

# %%
"""
# Base Balbino - Ultima
"""

# %%
# -*- coding: utf-8 -*-
"""
Este script realiza uma análise completa de modelos de classificação, com foco
em métricas detalhadas e na interpretação dos erros de classificação.

Novas Funcionalidades Adicionadas:
1.  **Métricas Gerais e por Classe com Desvio Padrão:** Na análise final,
    calcula e exibe a média e o desvio padrão para a acurácia geral,
    F1-score geral, e para precisão, recall e F1-score de cada classe.
2.  **Análise Completa para Todos os Modelos:** A análise geral no dataset
    completo (com gráficos, relatórios e métricas detalhadas) é agora
    realizada para TODOS os algoritmos.
3.  **Análise Profunda Apenas para o Melhor Modelo:** A geração de gráficos de
    divergência e a exportação de CSVs de erros são feitas apenas para o
    modelo com melhor performance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# OPF
try:
    from opfython.models import SupervisedOPF
    from opfython.stream import loader
except ImportError:
    SupervisedOPF = None
    loader = None

from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

# --- 1. FUNÇÕES AUXILIARES PARA ANÁLISE ---


class OPFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, distance='euclidean'):
        self.distance = distance
        self.model = None

    def fit(self, X, y):
        if SupervisedOPF is None:
            raise ImportError(
                "opfython não está disponível neste ambiente. "
                "Instale opfython em uma versão de Python compatível (<3.10)."
            )
        # Garante que X e y são arrays numpy
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        self.model = SupervisedOPF(distance=self.distance)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {'distance': self.distance}

    def set_params(self, **params):
        self.distance = params.get('distance', self.distance)
        return self

def analisar_e_exportar_todos_erros(y_true, y_pred, X_df, cm, target_names, model_name, n_top_features=3):
    """
    Analisa TODOS os erros da matriz de confusão, exibe uma tabela com os valores
    de divergência, plota as distribuições e SALVA os registros incorretos em CSV.
    """
    print("\n" + "#"*40)
    print(f"INICIANDO ANÁLISE PROFUNDA DE ERROS PARA O MELHOR MODELO: {model_name.upper()}")
    print("#" * 40)

    diretorio_erros = f'analise_de_erros_{model_name.replace(" ", "_")}'
    if not os.path.exists(diretorio_erros):
        os.makedirs(diretorio_erros)

    for true_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if true_idx == pred_idx:
                continue
            n_erros = cm[true_idx, pred_idx]
            if n_erros == 0:
                continue

            true_class = target_names[true_idx]
            pred_class = target_names[pred_idx]
            print(f"\n>> Analisando erro: '{true_class}' previsto como '{pred_class}' ({n_erros} ocorrências)")

            indices_acertos_classe_verdadeira = (y_true == true_idx) & (y_pred == true_idx)
            indices_erros_atuais = (y_true == true_idx) & (y_pred == pred_idx)
            df_acertos = X_df[indices_acertos_classe_verdadeira]
            df_erros = X_df[indices_erros_atuais].copy()

            df_erros['classe_verdadeira'] = true_class
            df_erros['classe_predita'] = pred_class
            nome_arquivo = f"{diretorio_erros}/erros_{true_class}_vs_{pred_class}.csv"
            df_erros.to_csv(nome_arquivo, index=False)
            print(f"   -> Registros de erro salvos em: '{nome_arquivo}'")

            if df_acertos.empty or df_erros.empty:
                print("   (Não há amostras suficientes de acertos ou erros para comparação gráfica.)")
                continue

            diferencas_media = (df_erros.drop(columns=['classe_verdadeira', 'classe_predita']).mean() - df_acertos.mean()).abs().sort_values(ascending=False)

            print("\n   --- Tabela de Divergência de Features (Top 5) ---")
            divergence_df = pd.DataFrame({
                'Feature': diferencas_media.head(5).index,
                'Divergência (Abs)': diferencas_media.head(5).values
            })
            print(divergence_df.to_string(index=False))

            features_para_plotar = diferencas_media.head(n_top_features).index

            for feature in features_para_plotar:
                plt.figure(figsize=(14, 8))
                if not df_acertos[feature].empty and df_acertos[feature].nunique() > 1:
                    sns.kdeplot(df_acertos[feature], label=f'Acertos (Verdadeiro: {true_class})', color='green', fill=True, alpha=0.5)
                    plt.axvline(df_acertos[feature].mean(), color='darkgreen', linestyle='--', label=f'Média Acertos ({df_acertos[feature].mean():.2f})')

                if not df_erros[feature].empty:
                    num_unique_errors = df_erros[feature].nunique()
                    if num_unique_errors > 1:
                        sns.kdeplot(df_erros[feature], label=f'Erros (Previsto como: {pred_class})', color='red', fill=True, alpha=0.5)
                        plt.axvline(df_erros[feature].mean(), color='darkred', linestyle='--', label=f'Média Erros ({df_erros[feature].mean():.2f})')
                    elif num_unique_errors == 1:
                        error_value = df_erros[feature].iloc[0]
                        plt.axvline(error_value, color='purple', linestyle='-', linewidth=3, label=f'Valor Único dos Erros ({error_value:.2f})')
                        y_pos = plt.gca().get_ylim()[1] * 0.5
                        plt.text(error_value, y_pos, '  <-- Todos os erros têm este valor', color='purple', ha='left', va='center', fontsize=10, weight='bold')

                main_title = f"Comparação de Distribuição da Feature '{feature}'"
                sub_title = f"Analisando o Erro: Classe Verdadeira '{true_class}' vs. Predição '{pred_class}'"
                plt.suptitle(main_title, fontsize=16, y=0.98)
                plt.title(sub_title, fontsize=12)
                plt.legend()
                plt.xlabel(f"Valor da Feature: {feature}")
                plt.ylabel("Densidade")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.figtext(0.5, -0.05, "Interpretação: Se as distribuições se sobrepõem muito, o modelo tem dificuldade em distinguir as classes com esta feature.", ha="center", fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# --- 2. SCRIPT PRINCIPAL DE PROCESSAMENTO ---

# --- Carregamento e Preparação Inicial ---
print("Carregando o dataset...")
try:
    df = pd.read_csv('edge_pls8_full.csv')
except FileNotFoundError:
    print("\nERRO: O arquivo 'edge_pls8_full.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o CSV: {e}")
    exit()

print(f"Shape do dataset: {df.shape}")

# --- Limpeza de Dados ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape do dataset após limpeza: {df.shape}")

# --- Separação e Codificação ---
# MODIFICAÇÃO: Removendo tanto o alvo (Attack_type) quanto a coluna 'attack_label' das features
X = df.drop(columns=['Attack_label'])
y_raw = df['Attack_label']

# Checa e remove a coluna 'frame.time' se ela existir
if 'frame.time' in X.columns:
    X = X.drop('frame.time', axis=1)

# Codifica colunas categóricas restantes em X
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codifica a variável alvo y
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_names = target_encoder.classes_
print(f"\nClasses no target: {len(np.unique(y))}: {target_names}")

# --- Amostragem para GridSearch ---
print("\nCriando uma amostra de 20% dos dados para a busca de hiperparâmetros...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# --- Definição dos Classificadores ---
classifiers = {
     'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'use_scaled_data': False
    },

    #'SVM': {
    #    'model': SVC(random_state=42, probability=True),
    #    'params': {
    #        'C': [0.1, 1, 10, 100],
    #        'kernel': ['linear', 'rbf', 'poly'],
    #        'gamma': ['scale', 'auto']
    #    },
    #    'use_scaled_data': True
    #},

    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        },
        'use_scaled_data': True
    },

    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'use_scaled_data': True
    },

    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        'use_scaled_data': True
    },

    #'OPF': {
     #     # Define com valor inicial válido
     #     'model': OPFClassifier(distance='euclidean'),
     #     'params': {
     #         'distance': ['additive_symmetric', 'average_euclidean', 'bhattacharyya', 'bray_curtis',
     #       'canberra', 'chebyshev', 'chi_squared', 'chord', 'clark', 'cosine',
     #       'dice', 'divergence', 'euclidean', 'gower', 'hamming', 'hassanat',
     #       'hellinger', 'jaccard', 'jeffreys', 'jensen', 'jensen_shannon',
     #       'k_divergence', 'kulczynski', 'kullback_leibler', 'lorentzian',
     #       'manhattan', 'matusita', 'mean_censored_euclidean', 'min_symmetric',
     #       'non_intersection', 'pearson', 'sangvi', 'soergel', 'squared_chord',
     #       'squared_euclidean', 'statistic', 'topsoe', 'vicis_symmetric1',
     #       'vicis_symmetric2', 'vicis_symmetric3', 'vicis_wave_hedges'
     #         ]
     #     },
     #     'use_scaled_data': True
     # }
}

# --- Execução da Análise na Amostra para encontrar os melhores parâmetros ---
skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

print("\n" + "="*80 + "\nINICIANDO BUSCA PELOS MELHORES HIPERPARÂMETROS (NA AMOSTRA)\n" + "="*80)

for name, clf_info in classifiers.items():
    print(f"\n{'-'*50}\nPROCESSANDO: {name}\n{'-'*50}")
    X_train_data = X_sample_scaled if clf_info['use_scaled_data'] else X_sample.values

    start_time = time.time()
    grid_search = GridSearchCV(estimator=clf_info['model'], param_grid=clf_info['params'], cv=skf_sample, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_sample)
    end_time = time.time()

    best_model_temp = grid_search.best_estimator_

    scoring_metrics = ['accuracy', 'f1_weighted']
    cv_scores = cross_validate(best_model_temp, X_train_data, y_sample, cv=skf_sample, scoring=scoring_metrics, n_jobs=-1)

    results[name] = {
        'best_model': best_model_temp,
        'processing_time': end_time - start_time,
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std()
    }
    print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
    print(f"Tempo de processamento: {results[name]['processing_time']:.2f} segundos")

# --- Resumo da Análise na Amostra ---
print("\n" + "="*80)
print("RESUMO FINAL DOS RESULTADOS (baseado na amostra de dados de 20%)")
print("="*80)

print("\n--- Métricas de Performance (Validação Cruzada na Amostra) ---")
print(f"{'Classificador':<22} {'Acurácia Média':<18} {'Acurácia (std)':<18} {'F1-Score Médio':<18} {'F1-Score (std)':<18}")
print("-" * 105)
for name, result in results.items():
    print(f"{name:<22} {result['cv_accuracy_mean']:.4f}              {result['cv_accuracy_std']:.4f}              {result['cv_f1_mean']:.4f}              {result['cv_f1_std']:.4f}")

print("\n--- Tempo de Processamento (GridSearch na Amostra) ---")
print(f"{'Classificador':<22} {'Tempo (segundos)':<20}")
print("-" * 45)
for name, result in results.items():
    print(f"{name:<22} {result['processing_time']:.2f}")

# --- Análise Final e Visualização PARA TODOS OS MODELOS ---
print("\n" + "="*80)
print("INICIANDO ANÁLISE DETALHADA NO DATASET COMPLETO PARA TODOS OS MODELOS")
print("="*80)

X_full_scaled = scaler.transform(X)
X_full_unscaled = X.values
skf_final = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
X_full_df_orig = X.copy()

final_predictions = {}

for model_name, result_info in results.items():
    print(f"\n\n{'#'*70}")
    print(f"### ANÁLISE GERAL PARA: {model_name.upper()} ###")
    print(f"{'#'*70}")

    model = result_info['best_model']
    X_full_data = X_full_scaled if classifiers[model_name]['use_scaled_data'] else X_full_unscaled

    # --- ANÁLISE DE MÉTRICAS GERAIS E POR CLASSE COM VALIDAÇÃO CRUZADA ---
    print("\nCalculando métricas gerais e por classe com validação cruzada...")

    # Dicionários para armazenar os scores de cada fold
    per_class_metrics_folds = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in target_names}
    accuracy_scores = []
    f1_weighted_scores = []

    for train_index, test_index in skf_final.split(X_full_data, y):
        X_train_fold, X_test_fold = X_full_data[train_index], X_full_data[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        report_fold = classification_report(y_test_fold, y_pred_fold, target_names=target_names, output_dict=True, zero_division=0)

        # Armazena scores gerais do fold
        accuracy_scores.append(report_fold['accuracy'])
        f1_weighted_scores.append(report_fold['weighted avg']['f1-score'])

        # Armazena scores por classe do fold
        for cls in target_names:
            if cls in report_fold:
                per_class_metrics_folds[cls]['precision'].append(report_fold[cls]['precision'])
                per_class_metrics_folds[cls]['recall'].append(report_fold[cls]['recall'])
                per_class_metrics_folds[cls]['f1-score'].append(report_fold[cls]['f1-score'])

    # --- Exibição das Métricas Gerais ---
    print("\n--- Métricas Gerais (Validação Cruzada no Dataset Completo) ---")
    print(f"Acurácia Média:       {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    print(f"F1-Score Ponderado Médio: {np.mean(f1_weighted_scores):.4f} (+/- {np.std(f1_weighted_scores):.4f})")

    # --- Exibição das Métricas por Classe ---
    print("\n--- Desempenho por Classe (Validação Cruzada no Dataset Completo) ---")
    header = f"{'Classe':<20} | {'Precisão':<18} | {'Recall':<18} | {'F1-Score':<18}"
    print(header)
    print("-" * len(header))
    for cls in target_names:
        prec_mean = np.mean(per_class_metrics_folds[cls]['precision'])
        prec_std = np.std(per_class_metrics_folds[cls]['precision'])
        rec_mean = np.mean(per_class_metrics_folds[cls]['recall'])
        rec_std = np.std(per_class_metrics_folds[cls]['recall'])
        f1_mean = np.mean(per_class_metrics_folds[cls]['f1-score'])
        f1_std = np.std(per_class_metrics_folds[cls]['f1-score'])
        print(f"{cls:<20} | {prec_mean:.4f} (+/- {prec_std:.4f}) | {rec_mean:.4f} (+/- {rec_std:.4f}) | {f1_mean:.4f} (+/- {f1_std:.4f})")


    # --- Gráfico de Erro de Treino vs Teste ---
    print("\nCalculando erros de treino e teste para análise de overfitting...")
    cv_results_errors = cross_validate(model, X_full_data, y, cv=skf_final, scoring='accuracy', return_train_score=True, n_jobs=-1)
    train_errors = 1 - cv_results_errors['train_score']
    test_errors = 1 - cv_results_errors['test_score']
    folds = range(1, skf_final.get_n_splits() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_errors, 'o-', color='blue', label='Erro de Treino')
    plt.plot(folds, test_errors, 'o-', color='red', label='Erro de Teste (Validação)')
    plt.title(f'Curva de Erro por Fold - {model_name}')
    plt.xlabel('Fold da Validação Cruzada')
    plt.ylabel('Taxa de Erro (1 - Acurácia)')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.show()

    # --- Matriz de Confusão ---
    print("\nGerando predições para a Matriz de Confusão...")
    y_pred_final = cross_val_predict(model, X_full_data, y, cv=skf_final, n_jobs=-1)
    final_predictions[model_name] = y_pred_final

    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Final (Validação Cruzada) - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

# --- ANÁLISE PROFUNDA APENAS PARA O MELHOR MODELO ---
best_model_name = max(results, key=lambda name: results[name]['cv_f1_mean'])
y_pred_best = final_predictions[best_model_name]
cm_best = confusion_matrix(y, y_pred_best)
print(cm_best)
#analisar_e_exportar_todos_erros(y, y_pred_best, X_full_df_orig, cm_best, target_names, best_model_name)

print("\n" + "="*80 + "\nANÁLISE CONCLUÍDA!\n" + "="*80)

# %%
