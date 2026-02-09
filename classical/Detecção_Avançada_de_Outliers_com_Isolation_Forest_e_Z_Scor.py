# %%
"""
#Detecção de Outliers

"""

# %%
# Detecção de Outliers - Edge-IIoTset Dataset (Versão Melhorada)
# Foco em Isolation Forest Aprimorado e Método Híbrido

## Célula 1: Instalação e Importação de Bibliotecas

# Instalação de bibliotecas (se necessário)
# !pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly

# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import jarque_bera, shapiro, probplot # Adicionado probplot aqui
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importar display se não estiver no ambiente Jupyter/Colab padrão
try:
    from IPython.display import display
except ImportError:
    print("IPython.display não encontrado. A função display() pode não funcionar.")
    def display(x): # Fallback simples para display
        print(x)


# Configurações de visualização
# plt.style.use('seaborn-v0_8') # Usar um estilo disponível ou 'default'
# Usar 'seaborn-v0_8-whitegrid' para um fundo branco com grades, que pode ser mais claro
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_palette("viridis")

print("✅ Bibliotecas carregadas com sucesso!")

## Célula 2: Carregamento e Análise Inicial dos Dados

# Carregamento dos dados Edge-IIoTset
df = pd.DataFrame() # Inicializar df para evitar NameError se o carregamento falhar
try:
    from google.colab import files
    print("Tentando carregar arquivo via Google Colab...")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
    print(f"Dataset '{file_name}' carregado via Colab.")
except ImportError:
    print("Google Colab 'files' não disponível. Tentando carregar localmente 'ML-EdgeIIoT-dataset.csv'")
    try:
        df = pd.read_csv('ML-EdgeIIoT-dataset.csv')
        print("Dataset 'ML-EdgeIIoT-dataset.csv' carregado localmente.")
    except FileNotFoundError:
        print("❌ Arquivo 'ML-EdgeIIoT-dataset.csv' não encontrado localmente. Por favor, verifique o caminho ou faça upload.")
        # df já está como DataFrame vazio
    except Exception as e_local:
        print(f"❌ Ocorreu um erro ao carregar o arquivo localmente: {e_local}")
        # df já está como DataFrame vazio
except Exception as e_colab:
    print(f"❌ Ocorreu um erro ao carregar o arquivo via Colab: {e_colab}")
    # df já está como DataFrame vazio


if not df.empty:
    # Informações básicas do dataset
    print(f"\n📊 Dataset carregado: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    print(f"📋 Colunas: {list(df.columns)}")
    print(f"💾 Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Identificar colunas numéricas e categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\n🔢 Colunas numéricas ({len(numeric_cols)}): {numeric_cols[:5]}..." if len(numeric_cols) > 5 else f"\n🔢 Colunas numéricas ({len(numeric_cols)}): {numeric_cols}")
    print(f"📝 Colunas categóricas ({len(categorical_cols)}): {categorical_cols}")

    # Análise de valores ausentes
    missing_analysis = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_analysis = missing_analysis[missing_analysis['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if not missing_analysis.empty:
        print(f"\n⚠️ Valores ausentes encontrados:")
        display(missing_analysis.head())
    else:
        print(f"\n✅ Nenhum valor ausente encontrado")
else:
    print("❌ Dataset vazio. A análise não pode prosseguir.")

## Célula 3: Análise Exploratória Avançada
if not df.empty:
    print("\n=== ANÁLISE EXPLORATÓRIA AVANÇADA ===\n")

    # Estatísticas descritivas detalhadas
    print("📊 Estatísticas descritivas:")
    if numeric_cols: # Verificar se há colunas numéricas
        stats_df = df[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        display(stats_df)

        # Análise de distribuições
        print(f"\n📈 Análise de Normalidade (amostra de até 5 variáveis):")
        sample_cols_norm = numeric_cols[:min(5, len(numeric_cols))] # Pegar até 5 colunas numéricas
        normality_results = {}

        for col_norm in sample_cols_norm:
            col_data_dropna = df[col_norm].dropna()
            if len(col_data_dropna) == 0:
                print(f"Coluna {col_norm} não possui dados após dropna(). Pulando teste de normalidade.")
                continue

            # Ajustar o tamanho da amostra para Shapiro-Wilk, se necessário, e garantir que não exceda o limite do teste
            sample_size_for_shapiro = min(5000 -1, len(col_data_dropna)) # Shapiro tem limite de 5000

            if len(col_data_dropna) >=3 and len(col_data_dropna) <= 5000 : # Shapiro requer pelo menos 3 amostras
                stat_norm, p_value_norm = shapiro(col_data_dropna.sample(sample_size_for_shapiro, random_state=42) if len(col_data_dropna) > sample_size_for_shapiro else col_data_dropna)
                test_name_norm = "Shapiro-Wilk"
            elif len(col_data_dropna) > 2: # Jarque-Bera requer pelo menos 2 amostras
                stat_norm, p_value_norm = jarque_bera(col_data_dropna)
                test_name_norm = "Jarque-Bera"
            else:
                print(f"Coluna {col_norm} tem menos de 3 amostras não-nulas. Pulando teste de normalidade.")
                continue


            normality_results[col_norm] = {
                'test': test_name_norm,
                'statistic': stat_norm,
                'p_value': p_value_norm,
                'normal': p_value_norm > 0.05,
                'skewness': df[col_norm].skew(),
                'kurtosis': df[col_norm].kurtosis()
            }

        if normality_results:
            normality_df = pd.DataFrame(normality_results).T
            print("Resultados dos testes de normalidade:")
            display(normality_df)
        else:
            print("Nenhuma coluna adequada para teste de normalidade.")


        # Matriz de correlação para identificar multicolinearidade
        # Selecionar apenas as primeiras 20 colunas numéricas para evitar matrizes muito grandes
        numeric_cols_for_corr = numeric_cols[:min(20, len(numeric_cols))]
        if len(numeric_cols_for_corr) > 1 : # Precisa de pelo menos 2 colunas para correlação
            corr_matrix = df[numeric_cols_for_corr].corr()

            # Encontrar correlações altas
            high_corr_pairs = []
            for i_corr in range(len(corr_matrix.columns)):
                for j_corr in range(i_corr+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i_corr, j_corr]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i_corr], corr_matrix.columns[j_corr], corr_matrix.iloc[i_corr, j_corr]))

            if high_corr_pairs:
                print(f"\n🔗 Correlações altas encontradas (>0.8) entre as primeiras {len(numeric_cols_for_corr)} features numéricas:")
                for col1_corr, col2_corr, corr_val in high_corr_pairs[:10]: # Mostrar apenas os 10 primeiros
                    print(f"   • {col1_corr} ↔ {col2_corr}: {corr_val:.3f}")
            else:
                print(f"\n🔗 Nenhuma correlação alta (>0.8) encontrada entre as primeiras {len(numeric_cols_for_corr)} features numéricas.")
        elif len(numeric_cols) > 20:
             print("\n🔗 Análise de correlação pulada (mais de 20 features numéricas). As primeiras 20 foram consideradas.")
        else:
             print("\n🔗 Análise de correlação pulada (menos de 2 features numéricas).")
    else:
        print("⚠️ Nenhuma coluna numérica para análise exploratória avançada.")
else:
    print("❌ Dataset vazio. Análise exploratória avançada não realizada.")

## Célula 4: Pré-processamento Avançado
df_processed = pd.DataFrame() # Inicializar
scaler_used = None
final_numeric_cols = [] # Colunas numéricas que realmente foram processadas e escaladas

if not df.empty:
    def preprocess_data(df_to_process, method='robust', handle_missing='median', remove_constant=True):
        """
        Pré-processamento avançado dos dados

        Parameters:
        - method: 'standard', 'robust', 'minmax'
        - handle_missing: 'median', 'mean', 'mode', 'drop'
        - remove_constant: remover colunas com variância zero
        """

        df_proc = df_to_process.copy()

        print(f"\n🔧 PRÉ-PROCESSAMENTO AVANÇADO")
        print(f"=" * 35)

        # 1. Tratamento de valores ausentes
        # Usar as colunas numéricas identificadas globalmente se disponíveis
        numeric_cols_clean = df_proc.select_dtypes(include=[np.number]).columns.tolist()


        if not numeric_cols_clean:
            print("⚠️ Nenhuma coluna numérica para tratar valores ausentes ou escalar.")
            return df_proc, None, [] # Retorna o df original e lista vazia de colunas processadas

        print(f"   Colunas numéricas inicialmente consideradas para pré-processamento: {len(numeric_cols_clean)}")

        if handle_missing == 'median':
            df_proc[numeric_cols_clean] = df_proc[numeric_cols_clean].fillna(df_proc[numeric_cols_clean].median())
        elif handle_missing == 'mean':
            df_proc[numeric_cols_clean] = df_proc[numeric_cols_clean].fillna(df_proc[numeric_cols_clean].mean())
        elif handle_missing == 'drop':
            # Guardar o índice antes de dropar linhas para poder alinhar com o df original depois, se necessário
            # df_proc_original_index = df_proc.index
            df_proc = df_proc.dropna(subset=numeric_cols_clean)
            print(f"   Linhas com NaNs em colunas numéricas foram removidas. Novo shape: {df_proc.shape}")
        # 'mode' é mais complexo para numéricas e geralmente não é a melhor escolha, então foi omitido por simplicidade.

        print(f"✅ Valores ausentes em colunas numéricas tratados usando: {handle_missing}")

        # 2. Remover colunas constantes
        if remove_constant:
            constant_cols_to_remove = []
            # Iterar sobre as colunas numéricas que ainda existem no df_proc
            current_numeric_in_df_proc = df_proc.select_dtypes(include=[np.number]).columns.tolist()
            for col_const in current_numeric_in_df_proc:
                # Certificar que a coluna ainda existe antes de verificar nunique
                if col_const in df_proc.columns and df_proc[col_const].nunique(dropna=False) <= 1: # Considerar NaNs como um valor único se existirem
                    constant_cols_to_remove.append(col_const)

            if constant_cols_to_remove:
                df_proc = df_proc.drop(columns=constant_cols_to_remove)
                print(f"🗑️  Removidas {len(constant_cols_to_remove)} colunas constantes: {constant_cols_to_remove}")
                # Atualizar a lista de colunas numéricas limpas após a remoção
                numeric_cols_clean = [col for col in numeric_cols_clean if col not in constant_cols_to_remove]


        # 3. Identificar e tratar valores infinitos
        inf_cols_treated = []
        # Re-identificar colunas numéricas no df_proc atual, pois algumas podem ter sido removidas
        current_numeric_cols_for_inf = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        for col_inf in current_numeric_cols_for_inf:
            if np.isinf(df_proc[col_inf].values).any():
                inf_cols_treated.append(col_inf)
                df_proc[col_inf] = df_proc[col_inf].replace([np.inf, -np.inf], np.nan)
                # Preencher NaNs resultantes da substituição de inf com a mediana da coluna
                df_proc[col_inf] = df_proc[col_inf].fillna(df_proc[col_inf].median()) # Poderia ser mean() também

        if inf_cols_treated:
            print(f"♾️  Tratados valores infinitos (substituídos por NaN e depois pela mediana) em {len(inf_cols_treated)} colunas: {inf_cols_treated}")

        # 4. Normalização/Padronização
        # Usar as colunas numéricas que restaram após limpeza de constantes e tratamento de infinitos
        numeric_cols_for_scaling = df_proc.select_dtypes(include=[np.number]).columns.tolist()

        scaler_instance = None
        scaler_name_str = "Nenhum"

        if not numeric_cols_for_scaling:
            print("⚠️ Nenhuma coluna numérica restante para normalização/padronização.")
            processed_cols_list = [] # Nenhuma coluna foi escalada
        else:
            if method == 'standard':
                scaler_instance = StandardScaler()
                scaler_name_str = "StandardScaler (Z-score)"
            elif method == 'robust':
                scaler_instance = RobustScaler()
                scaler_name_str = "RobustScaler (mediana/IQR)"
            elif method == 'minmax':
                scaler_instance = MinMaxScaler()
                scaler_name_str = "MinMaxScaler (0-1)"
            else:
                print(f"⚠️ Método de scaling '{method}' não reconhecido. Nenhuma normalização aplicada.")
                scaler_name_str = "Nenhum (método não reconhecido)"
                processed_cols_list = numeric_cols_for_scaling # Colunas que passaram pelos outros passos, mas não foram escaladas

            if scaler_instance:
                # Assegurar que não há NaNs antes de escalar, pois alguns scalers podem falhar
                # Esta etapa é crucial, especialmente se os passos anteriores de fillna não cobriram tudo
                # ou se novas colunas foram introduzidas (o que não é o caso aqui)
                df_proc[numeric_cols_for_scaling] = df_proc[numeric_cols_for_scaling].fillna(df_proc[numeric_cols_for_scaling].median())

                # Checar se há alguma coluna completamente NaN antes de escalar
                cols_all_nan = df_proc[numeric_cols_for_scaling].isnull().all()
                if cols_all_nan.any():
                    print(f"⚠️ As seguintes colunas são completamente NaN e não serão escaladas: {cols_all_nan[cols_all_nan].index.tolist()}")
                    # Remover colunas que são todas NaN da lista de escalonamento
                    numeric_cols_for_scaling = [col for col in numeric_cols_for_scaling if not cols_all_nan[col]]


                if not numeric_cols_for_scaling: # Se todas as colunas numéricas se tornaram NaN
                     print("⚠️ Todas as colunas numéricas para scaling são NaN. Pular scaling.")
                     processed_cols_list = []
                else:
                    df_proc_scaled_values = scaler_instance.fit_transform(df_proc[numeric_cols_for_scaling])
                    df_proc_scaled_df = pd.DataFrame(df_proc_scaled_values, index=df_proc.index, columns=numeric_cols_for_scaling)
                    df_proc[numeric_cols_for_scaling] = df_proc_scaled_df
                    print(f"📊 Dados numéricos normalizados usando: {scaler_name_str} em {len(numeric_cols_for_scaling)} colunas.")
                    processed_cols_list = numeric_cols_for_scaling # Estas são as colunas que foram de fato escaladas
            else: # Se scaler_instance não foi criado (e.g. método não reconhecido)
                processed_cols_list = numeric_cols_for_scaling


        print(f"📋 Shape final do pré-processamento: {df_proc.shape}")
        # Retorna as colunas que foram efetivamente submetidas ao scaling (ou que teriam sido se um scaler fosse aplicado)
        return df_proc, scaler_instance, processed_cols_list


    # Aplicar o pré-processamento
    # É importante que `final_numeric_cols` reflita as colunas que REALMENTE existem em df_processed e foram escaladas.
    df_processed, scaler_used, final_numeric_cols = preprocess_data(
        df.copy(), # Passar uma cópia para não modificar o df original
        method='robust',
        handle_missing='median',
        remove_constant=True
    )
    # Verificar se df_processed não está vazio e se final_numeric_cols também não está
    if df_processed.empty or not final_numeric_cols:
        print("❌ Pré-processamento resultou em DataFrame vazio ou sem colunas numéricas válidas. Análises subsequentes podem falhar.")

else:
    print("❌ Dataset original vazio. Pré-processamento não realizado.")


## Célula 5: Isolation Forest Avançado
best_model = None # Modelo IF treinado
best_results = {}   # Dicionário com resultados da melhor config do IF (scores, mask, etc.)

if not df_processed.empty and final_numeric_cols: # Garantir que há dados e colunas para usar
    class AdvancedIsolationForest:
        def __init__(self):
            self.models = {}
            self.results = {}
            self.best_model = None
            self.best_config_key = None

        def fit_multiple_configs(self, X_df, feature_names_list): # X_df é o dataframe, feature_names_list são as colunas a usar
            if not isinstance(X_df, pd.DataFrame):
                print("❌ X_df deve ser um DataFrame pandas.")
                return None, {}
            if X_df.empty:
                print("❌ X_df (features) está vazio.")
                return None, {}
            if not feature_names_list:
                print("❌ Lista de nomes de features (feature_names_list) está vazia.")
                return None, {}

            # Selecionar apenas as colunas especificadas e garantir que não há NaNs
            X_values = X_df[feature_names_list].dropna().values
            if X_values.shape[0] == 0:
                print(f"❌ X_df após selecionar features '{feature_names_list}' e dropar NaNs está vazio. Não é possível treinar.")
                return None, {}
            if X_values.ndim == 1: # Se restar apenas uma feature, reshape para 2D
                X_values = X_values.reshape(-1,1)


            configs = [
                {'contamination': 0.05, 'n_estimators': 100, 'max_samples': 'auto', 'random_state': 42},
                {'contamination': 0.08, 'n_estimators': 150, 'max_samples': min(256, X_values.shape[0]), 'random_state': 42}, # max_samples não pode ser > n_samples
                {'contamination': 0.10, 'n_estimators': 200, 'max_samples': 'auto', 'random_state': 42},
                {'contamination': 0.12, 'n_estimators': 100, 'max_samples': min(512, X_values.shape[0]), 'random_state': 42},
                {'contamination': 0.15, 'n_estimators': 250, 'max_samples': 'auto', 'random_state': 42}
            ]
            print(f"\n🤖 ISOLATION FOREST AVANÇADO (sobre df_processed)\n" + "=" * 48 + f"\n📊 Testando {len(configs)} configurações em {X_values.shape[0]} amostras e {X_values.shape[1]} features...\n")

            for i_conf, config_if in enumerate(configs):
                # Ajustar max_samples se for string 'auto' ou maior que o número de amostras
                if isinstance(config_if['max_samples'], str) and config_if['max_samples'] == 'auto':
                    current_max_samples = min(256, X_values.shape[0]) # Padrão do sklearn para 'auto' é min(256, n_samples)
                elif isinstance(config_if['max_samples'], int):
                    current_max_samples = min(config_if['max_samples'], X_values.shape[0])
                else: # Fallback
                    current_max_samples = min(256, X_values.shape[0])

                # Garantir que n_estimators é pelo menos 1
                current_n_estimators = max(1, config_if['n_estimators'])

                model_if = IsolationForest(
                    n_estimators=current_n_estimators,
                    max_samples=current_max_samples,
                    contamination=config_if['contamination'],
                    random_state=config_if['random_state'],
                    bootstrap=False # Default é False, mas pode ser experimentado
                )
                try:
                    predictions_if = model_if.fit_predict(X_values)
                    scores_if = model_if.decision_function(X_values) # Scores são negativos para outliers, mais negativos = mais anômalo
                except ValueError as e_if_fit:
                    print(f"   Erro ao treinar config {i_conf+1} (cont={config_if['contamination']}): {e_if_fit}. Pulando esta config.")
                    continue


                outliers_mask_if = predictions_if == -1
                n_outliers_if = sum(outliers_mask_if)
                outlier_percentage_if = (n_outliers_if / len(X_values)) * 100 if len(X_values) > 0 else 0
                separation_quality_if = self._calculate_separation_quality(scores_if, outliers_mask_if)

                self.models[f"config_{i_conf+1}"] = model_if
                self.results[f"config_{i_conf+1}"] = {
                    'config': config_if,
                    'predictions': predictions_if, # Array de 1s e -1s
                    'scores': scores_if,           # Array de scores de anomalia
                    'outliers_mask': outliers_mask_if, # Array booleano (True para outlier)
                    'n_outliers': n_outliers_if,
                    'outlier_percentage': outlier_percentage_if,
                    'separation_quality': separation_quality_if,
                    'score_std': np.std(scores_if) if len(scores_if) > 0 else 0
                }
                print(f"⚙️  Config {i_conf+1}: cont={config_if['contamination']}, n_est={current_n_estimators}, max_s={current_max_samples}")
                print(f"   📈 Outliers: {n_outliers_if:,} ({outlier_percentage_if:.2f}%), Qualidade Sep: {separation_quality_if:.3f}\n")

            if self.results: # Se algum modelo foi treinado com sucesso
                # Critério de seleção: maior qualidade de separação, desempatando por menor std nos scores (mais confiança)
                self.best_config_key = max(self.results.keys(), key=lambda k: (self.results[k]['separation_quality'], -self.results[k]['score_std']))
                self.best_model = self.models[self.best_config_key]
                best_data_if = self.results[self.best_config_key]
                print(f"🏆 MELHOR CONFIG (IF): {self.best_config_key}\n   Outliers: {best_data_if['n_outliers']:,} ({best_data_if['outlier_percentage']:.2f}%), Qualidade: {best_data_if['separation_quality']:.3f}\n   Config: {best_data_if['config']}")
                return self.best_model, best_data_if # Retorna o modelo e o dict de resultados da melhor config

            print("⚠️ Nenhuma configuração do Isolation Forest foi treinada com sucesso.");
            return None, {} # Retorna None se nenhum modelo foi treinado

        def _calculate_separation_quality(self, scores_sep, outliers_mask_sep):
            if sum(outliers_mask_sep) == 0 or sum(~outliers_mask_sep) == 0 or len(scores_sep) == 0: return 0.0 # Evitar divisão por zero ou dados insuficientes
            normal_s_sep = scores_sep[~outliers_mask_sep]
            outlier_s_sep = scores_sep[outliers_mask_sep]
            if len(normal_s_sep) == 0 or len(outlier_s_sep) == 0: return 0.0

            mean_d_sep = abs(np.mean(normal_s_sep) - np.mean(outlier_s_sep))
            # Variância agrupada (pooled variance) para calcular desvio padrão agrupado
            var_normal = np.var(normal_s_sep, ddof=1) if len(normal_s_sep) > 1 else 0
            var_outlier = np.var(outlier_s_sep, ddof=1) if len(outlier_s_sep) > 1 else 0

            # Evitar divisão por zero se uma das variâncias for zero e o tamanho do grupo for 1
            if (len(normal_s_sep) == 1 and var_normal == 0) or \
               (len(outlier_s_sep) == 1 and var_outlier == 0) :
                 # Se ambos os grupos têm tamanho 1, não se pode calcular pooled_std de forma significativa
                 if len(normal_s_sep) == 1 and len(outlier_s_sep) == 1:
                     pooled_s_sep = 1e-8 # Pequeno valor para evitar divisão por zero, mas indica baixa confiança
                 # Se um grupo tem tamanho 1 e variância 0, usar std do outro grupo
                 elif len(normal_s_sep) == 1 and var_normal == 0 and len(outlier_s_sep) > 1:
                     pooled_s_sep = np.sqrt(var_outlier) if var_outlier > 0 else 1e-8
                 elif len(outlier_s_sep) == 1 and var_outlier == 0 and len(normal_s_sep) > 1:
                     pooled_s_sep = np.sqrt(var_normal) if var_normal > 0 else 1e-8
                 else: # Fallback
                     pooled_s_sep = np.sqrt((var_normal + var_outlier) / 2) if (var_normal + var_outlier) > 0 else 1e-8
            else: # Caso geral
                 pooled_s_sep = np.sqrt((var_normal + var_outlier) / 2) if (var_normal + var_outlier) > 0 else 1e-8


            return mean_d_sep / (pooled_s_sep + 1e-8) # Adicionar epsilon para evitar divisão por zero

        def get_feature_importance(self, X_df_feat_imp, feature_names_list_feat_imp, trained_model_feat_imp):
            # X_df_feat_imp é o DataFrame (geralmente df_processed)
            # feature_names_list_feat_imp é a lista de colunas usadas para treinar (final_numeric_cols)
            # trained_model_feat_imp é o modelo IF treinado (self.best_model)
            if trained_model_feat_imp is None: print("❌ Modelo não treinado!"); return None
            if X_df_feat_imp.empty or not feature_names_list_feat_imp: print("❌ X_df_feat_imp ou features vazias."); return None

            X_val_feat_imp = X_df_feat_imp[feature_names_list_feat_imp].values
            if X_val_feat_imp.shape[0] == 0 : print("❌ X_val_feat_imp (dados) está vazio após selecionar features."); return None

            original_s_feat_imp = trained_model_feat_imp.decision_function(X_val_feat_imp)
            importances_feat_imp = {}

            for i_feat_imp, feature_name_feat_imp in enumerate(feature_names_list_feat_imp):
                X_perm_feat_imp = X_val_feat_imp.copy()
                # Permutar apenas a coluna atual
                X_perm_feat_imp[:, i_feat_imp] = np.random.permutation(X_perm_feat_imp[:, i_feat_imp])
                permuted_scores_feat_imp = trained_model_feat_imp.decision_function(X_perm_feat_imp)
                importances_feat_imp[feature_name_feat_imp] = np.mean(np.abs(original_s_feat_imp - permuted_scores_feat_imp))

            max_imp_feat_imp = max(importances_feat_imp.values()) if importances_feat_imp else 0
            # Normalizar importâncias
            normalized_importances = {
                k_imp: (v_imp / max_imp_feat_imp if max_imp_feat_imp > 0 else 0.0)
                for k_imp, v_imp in importances_feat_imp.items()
            }
            return dict(sorted(normalized_importances.items(), key=lambda x: x[1], reverse=True))

    advanced_if = AdvancedIsolationForest()
    # Passar df_processed e a lista de colunas numéricas finais para o fit
    best_model, best_results = advanced_if.fit_multiple_configs(df_processed, final_numeric_cols)
    # best_results agora contém 'predictions', 'scores', 'outliers_mask' etc.
    # Essas chaves são baseadas nos dados de df_processed[final_numeric_cols]
else:
    print("❌ df_processed vazio ou sem final_numeric_cols. Isolation Forest Avançado não aplicado.")

## Célula 6: Análise de Importância das Features
feature_importance = None # Dicionário com importâncias
if best_model and not df_processed.empty and final_numeric_cols:
    print("\n🔍 ANÁLISE DE IMPORTÂNCIA DAS FEATURES (baseado no melhor IF e df_processed)"); print("=" * 70)
    feature_importance = advanced_if.get_feature_importance(df_processed, final_numeric_cols, best_model)

    if feature_importance:
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        print("📊 Top 10 Features mais importantes:")
        display(importance_df.head(10))

        # Visualização
        fig_imp, axes_imp = plt.subplots(1, 2, figsize=(18, 7)) # Aumentar um pouco o tamanho
        top_features_plot_imp = importance_df.head(15)

        sns.barplot(x='Importance', y='Feature', data=top_features_plot_imp, ax=axes_imp[0], palette='coolwarm')
        axes_imp[0].set_xlabel('Importância Normalizada', fontsize=12)
        axes_imp[0].set_ylabel('Feature', fontsize=12)
        axes_imp[0].set_title('Top 15 Features - Detecção de Outliers (IF)', fontsize=14, fontweight='bold')
        axes_imp[0].tick_params(axis='y', labelsize=10)


        axes_imp[1].hist(importance_df['Importance'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes_imp[1].set_xlabel('Importância Normalizada', fontsize=12)
        axes_imp[1].set_ylabel('Frequência', fontsize=12)
        axes_imp[1].set_title('Distribuição das Importâncias das Features', fontsize=14, fontweight='bold')
        if not importance_df.empty:
            axes_imp[1].axvline(x=np.mean(importance_df['Importance']), color='red', linestyle='--', label=f'Média: {np.mean(importance_df["Importance"]):.2f}')
        axes_imp[1].legend(fontsize=10)

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Não foi possível calcular importância das features.")
else:
    print("⚠️ Modelo não treinado ou dados (df_processed, final_numeric_cols) insuficientes para análise de importância.")


## Célula 7: Visualizações Avançadas
# Estas visualizações são baseadas em df_processed e os resultados do IF (best_results)
if best_results and 'outliers_mask' in best_results and not df_processed.empty and final_numeric_cols:

    def create_advanced_visualizations(
        df_original_for_viz,      # df (original, para timeline se possível)
        df_proc_for_viz,          # df_processed (para PCA, scores)
        results_dict_viz,         # best_results (contém outliers_mask, scores ALINHADOS com df_proc_for_viz)
        numeric_cols_in_proc_viz, # final_numeric_cols
        feat_importance_dict_viz=None # Opcional
        ):

        print("\n📊 VISUALIZAÇÕES AVANÇADAS (baseadas em df_processed e resultados do IF)"); print("=" * 70)
        # Extrair dados de results_dict_viz. Estes estão alinhados com df_proc_for_viz.
        outliers_mask_viz_proc = results_dict_viz.get('outliers_mask', np.array([])) # Máscara do IF em df_processed
        scores_viz_proc = results_dict_viz.get('scores', np.array([]))             # Scores do IF em df_processed

        if len(outliers_mask_viz_proc) == 0 or len(scores_viz_proc) == 0:
            print("⚠️ Máscara de outliers ou scores não disponíveis em results_dict_viz. Visualizações avançadas puladas."); return
        if len(outliers_mask_viz_proc) != len(df_proc_for_viz):
             print(f"⚠️ Desalinhamento: máscara de outliers ({len(outliers_mask_viz_proc)}) vs df_processed ({len(df_proc_for_viz)}). Visualizações podem ser imprecisas."); return


        fig_plotly = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribuição dos Scores de Anomalia (IF em df_processed)',
                'PCA 3D dos Dados Processados (Colorido por Score IF)',
                'Contagem de Classes (Normal vs. Outlier IF em df_processed)',
                'Timeline de Detecção de Outliers (se coluna de tempo existir em df_original)'
            ),
            specs=[[{}, {"type": "scatter3d"}], [{}, {}]] # Especificar tipo 3D para o subplot PCA
        )

        # 1. Histograma dos Scores de Anomalia
        fig_plotly.add_trace(
            go.Histogram(x=scores_viz_proc[~outliers_mask_viz_proc], name='Normal (IF)', opacity=0.75, marker_color='green', nbinsx=50),
            row=1, col=1
        )
        fig_plotly.add_trace(
            go.Histogram(x=scores_viz_proc[outliers_mask_viz_proc], name='Outlier (IF)', opacity=0.75, marker_color='red', nbinsx=50),
            row=1, col=1
        )
        fig_plotly.update_xaxes(title_text="Score de Anomalia (IF)", row=1, col=1)
        fig_plotly.update_yaxes(title_text="Contagem", row=1, col=1)


        # 2. PCA 3D
        if len(numeric_cols_in_proc_viz) >= 3 and not df_proc_for_viz[numeric_cols_in_proc_viz].empty:
            try:
                pca = PCA(n_components=3, random_state=42)
                # Usar apenas as colunas numéricas que foram processadas
                pca_result = pca.fit_transform(df_proc_for_viz[numeric_cols_in_proc_viz])

                fig_plotly.add_trace(
                    go.Scatter3d(
                        x=pca_result[:,0], y=pca_result[:,1], z=pca_result[:,2],
                        mode='markers',
                        marker=dict(
                            color=scores_viz_proc, # Colorir por scores do IF (de df_processed)
                            colorscale='Viridis',  # Escala de cores
                            size=3,
                            opacity=0.6,
                            colorbar=dict(title="Score Anomalia (IF)")
                        ),
                        customdata=outliers_mask_viz_proc, # Passar a máscara do IF para hover
                        hovertemplate="<b>Outlier (IF)</b>: %{customdata}<br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>PCA3: %{z:.2f}<extra></extra>",
                        name='Dados Processados (PCA)'
                    ),
                    row=1, col=2
                )
                fig_plotly.update_layout(scene=dict(xaxis_title='PCA Comp. 1', yaxis_title='PCA Comp. 2', zaxis_title='PCA Comp. 3'))
            except Exception as e_pca:
                print(f"Erro ao gerar PCA 3D: {e_pca}")
                fig_plotly.add_annotation(text=f"Erro PCA: {e_pca}", row=1, col=2, showarrow=False, font=dict(color="red"))

        else:
            fig_plotly.add_annotation(text="PCA 3D requer >=3 features numéricas em df_processed.", row=1, col=2, showarrow=False)

        # 3. Contagem de Classes (Normal vs Outlier IF)
        counts_viz = pd.Series(outliers_mask_viz_proc).value_counts()
        fig_plotly.add_trace(
            go.Bar(
                x=['Normal (IF)','Outliers (IF)'],
                y=[counts_viz.get(False,0), counts_viz.get(True,0)],
                marker_color=['green','red'], name='Contagem Classes (IF)',
                text=[counts_viz.get(False,0), counts_viz.get(True,0)], textposition='auto'
                ),
            row=2, col=1
        )
        fig_plotly.update_yaxes(title_text="Número de Amostras", row=2, col=1)


        # 4. Timeline de Detecção (usando df_original_for_viz e alinhando a máscara do IF)
        timeline_added = False
        time_cols_in_original = [c for c in df_original_for_viz.columns if 'time' in c.lower() or 'date' in c.lower()]

        if time_cols_in_original:
            try:
                # Criar uma cópia do DataFrame original para não modificá-lo
                df_time_viz = df_original_for_viz.copy()
                time_col_to_use = time_cols_in_original[0] # Usar a primeira coluna de tempo encontrada
                df_time_viz[time_col_to_use] = pd.to_datetime(df_time_viz[time_col_to_use], errors='coerce')
                df_time_viz = df_time_viz.dropna(subset=[time_col_to_use]) # Remover linhas onde a data não pôde ser convertida

                if not df_time_viz.empty:
                    # Alinhar a outliers_mask_viz_proc (que é de df_proc_for_viz) com df_time_viz (que é derivado de df_original_for_viz)
                    # Isso só funciona bem se df_proc_for_viz manteve os índices originais de df_original_for_viz
                    # ou se df_proc_for_viz é um subconjunto com os mesmos índices.
                    if df_proc_for_viz.index.equals(df_original_for_viz.index) or df_proc_for_viz.index.isin(df_original_for_viz.index).all():
                        # Criar uma Series da máscara com o índice de df_proc_for_viz
                        mask_series_on_proc_index = pd.Series(outliers_mask_viz_proc, index=df_proc_for_viz.index)
                        # Reindexar para o índice de df_time_viz (que é o mesmo de df_original_for_viz após tratamento de data)
                        # fill_value=False significa que índices em df_time_viz não presentes em mask_series_on_proc_index serão False (não outlier)
                        aligned_mask_for_time_viz = mask_series_on_proc_index.reindex(df_time_viz.index, fill_value=False).values
                        df_time_viz['is_outlier_if'] = aligned_mask_for_time_viz


                        # Agrupar por hora (ou outra frequência apropriada)
                        # Certifique-se que a coluna de tempo é o índice ou especifique com 'on='
                        if not pd.api.types.is_datetime64_any_dtype(df_time_viz[time_col_to_use]):
                             print("Coluna de tempo não é datetime após conversão. Pulando timeline.")
                        else:
                            # Ordenar por tempo antes de agrupar
                            df_time_viz = df_time_viz.sort_values(by=time_col_to_use)
                            grouped_time_viz = df_time_viz.groupby(pd.Grouper(key=time_col_to_use, freq='H'))['is_outlier_if'].sum().reset_index()

                            if not grouped_time_viz.empty:
                                fig_plotly.add_trace(
                                    go.Scatter(x=grouped_time_viz[time_col_to_use], y=grouped_time_viz['is_outlier_if'], mode='lines+markers', name='Outliers (IF)/Hora'),
                                    row=2, col=2
                                )
                                fig_plotly.update_xaxes(title_text=f"Tempo ({time_col_to_use} - Agrupado por Hora)", row=2, col=2)
                                fig_plotly.update_yaxes(title_text="Contagem de Outliers (IF)", row=2, col=2)
                                timeline_added = True
                            else: print("Agrupamento temporal resultou em dados vazios para timeline.")
                    else:
                        print("Índices de df_processed e df_original não se alinham para a timeline.")
                else: print("DataFrame para timeline vazio após tratamento da coluna de tempo.")
            except Exception as e_timeline:
                print(f"Erro ao gerar visualização de timeline: {e_timeline}")
        else:
            print("Nenhuma coluna de tempo/data encontrada no DataFrame original para a timeline.")

        if not timeline_added:
            fig_plotly.add_annotation(text="Timeline de outliers não disponível\n(sem coluna de tempo ou erro no processamento).", row=2, col=2, showarrow=False)

        fig_plotly.update_layout(
            height=900, # Aumentar altura para melhor visualização
            title_text="<b>Dashboard de Detecção de Outliers (Resultados do Isolation Forest em Dados Processados)</b>",
            title_font_size=20,
            legend_title_text='Legenda',
            showlegend=True
        )
        fig_plotly.show()

        # Visualização Matplotlib simples (distribuição dos scores)
        fig_mpl, ax_mpl = plt.subplots(1,1,figsize=(10,6))
        if len(scores_viz_proc)>0:
            sns.histplot(scores_viz_proc[~outliers_mask_viz_proc], color="green", label='Normal (IF)', kde=True, ax=ax_mpl, stat="density", element="step")
            sns.histplot(scores_viz_proc[outliers_mask_viz_proc], color="red", label='Outlier (IF)', kde=True, ax=ax_mpl, stat="density", element="step")
            ax_mpl.set_title("Distribuição dos Scores de Anomalia (IF em df_processed) - Matplotlib", fontsize=14, fontweight='bold')
            ax_mpl.set_xlabel("Score de Anomalia (IF)", fontsize=12)
            ax_mpl.set_ylabel("Densidade", fontsize=12)
            ax_mpl.legend()
            plt.show()
        else:
            plt.close(fig_mpl) # Fechar a figura se não houver dados

    # Chamada da função de visualização avançada:
    # Passar df (original), df_processed, best_results (do IF), e final_numeric_cols
    create_advanced_visualizations(df, df_processed, best_results, final_numeric_cols, feature_importance)
else:
    print("⚠️ Visualizações avançadas (Célula 7) puladas: dados insuficientes (best_results, df_processed, ou final_numeric_cols).")


## Célula 8: Comparação com Z-Score Otimizado e Método Híbrido
zscore_comparison_results = {} # Resultados do Z-Score (dict por threshold)
z_score_feature_counts_df = pd.DataFrame() # DataFrame com contagem de features excedendo Z por amostra
hybrid_outliers_mask = np.array([]) # Máscara final do método híbrido (alinhada com df_processed)

if not df_processed.empty and final_numeric_cols: # Requer df_processed e as colunas numéricas usadas

    def optimized_zscore_detection(df_z_proc, numeric_cols_z_proc, threshold_range_z=[2.5, 3.0, 3.5, 4.0]):
        # df_z_proc é o df_processed
        # numeric_cols_z_proc é final_numeric_cols
        if df_z_proc.empty or not numeric_cols_z_proc:
            print("⚠️ Z-Score Otimizado: df_processed ou final_numeric_cols vazios."); return {}, pd.DataFrame()

        print("\n📊 Z-SCORE OTIMIZADO (sobre df_processed)"); print("=" * 48)
        results_z_dict = {}
        # Inicializar DataFrame para contagem de features por amostra com índice de df_z_proc
        z_counts_per_sample_df = pd.DataFrame(0, index=df_z_proc.index, columns=threshold_range_z)

        for thresh_z_val in threshold_range_z:
            # Máscara para identificar se QUALQUER feature excedeu o Z-score para esta amostra
            overall_outlier_mask_for_thresh_z = pd.Series(False, index=df_z_proc.index)
            # Contagem de quantas features excederam Z para cada amostra
            feature_exceed_count_for_thresh_z = pd.Series(0, index=df_z_proc.index)

            for col_z_proc in numeric_cols_z_proc:
                if col_z_proc in df_z_proc.columns:
                    data_col_z = df_z_proc[col_z_proc].dropna() # Usar dados já limpos e escalados
                    if data_col_z.empty or data_col_z.std() == 0: # Pular se coluna vazia ou sem variação
                        continue

                    z_values = np.abs(stats.zscore(data_col_z))
                    # Identificar os ÍNDICES das amostras que são outliers para ESTA COLUNA
                    outlier_indices_for_col = data_col_z[z_values > thresh_z_val].index

                    # Marcar essas amostras como outliers no geral para este threshold
                    overall_outlier_mask_for_thresh_z.loc[outlier_indices_for_col] = True
                    # Incrementar a contagem de features para essas amostras
                    feature_exceed_count_for_thresh_z.loc[outlier_indices_for_col] += 1
                else:
                    print(f"   Aviso Z-Score: Coluna {col_z_proc} não encontrada em df_processed.")


            z_counts_per_sample_df[thresh_z_val] = feature_exceed_count_for_thresh_z
            num_unique_outliers_z = overall_outlier_mask_for_thresh_z.sum()
            perc_outliers_z = (num_unique_outliers_z / len(df_z_proc)) * 100 if len(df_z_proc) > 0 else 0

            results_z_dict[thresh_z_val] = {
                'total_outliers': num_unique_outliers_z,
                'percentage': perc_outliers_z,
                'outlier_mask_z_score': overall_outlier_mask_for_thresh_z # Máscara booleana para este threshold
            }
            print(f"   Thresh Z={thresh_z_val}: {num_unique_outliers_z:,} outliers ({perc_outliers_z:.2f}%)")
        return results_z_dict, z_counts_per_sample_df

    zscore_comparison_results, z_score_feature_counts_df = optimized_zscore_detection(df_processed, final_numeric_cols)

    # Método Híbrido (IF + Z-Score)
    # Usa a 'outliers_mask' do best_results do IF (que está alinhada com df_processed)
    # e z_score_feature_counts_df (que também está alinhada com df_processed)
    if best_results and 'outliers_mask' in best_results and not z_score_feature_counts_df.empty:
        print("\n🧬 MÉTODO HÍBRIDO (IF + Z-Score, sobre df_processed)"); print("=" * 55)

        mask_if_from_best_results = best_results['outliers_mask'] # Alinhada com df_processed

        # Parâmetros para o método híbrido
        HYBRID_Z_SCORE_THRESHOLD = 3.0 # Threshold do Z-score a ser usado
        MIN_Z_SCORE_FEATURES_COUNT = 1 # Mínimo de features que devem exceder o Z-score

        # Verificar se o threshold do Z-score escolhido foi calculado
        if HYBRID_Z_SCORE_THRESHOLD not in z_score_feature_counts_df.columns:
            print(f"⚠️ Threshold Z-Score para Híbrido ({HYBRID_Z_SCORE_THRESHOLD}) não foi pré-calculado em z_score_feature_counts_df.")
            # Tentar usar o primeiro threshold disponível como fallback, ou pular o híbrido
            if not z_score_feature_counts_df.columns.empty:
                fallback_z_thresh = z_score_feature_counts_df.columns[0]
                print(f"   Usando fallback Z-Score threshold: {fallback_z_thresh} para o método híbrido.")
                z_score_counts_for_hybrid = z_score_feature_counts_df[fallback_z_thresh]
            else:
                print("   Não há thresholds Z-Score disponíveis. Método Híbrido não pode ser aplicado.")
                z_score_counts_for_hybrid = pd.Series(0, index=df_processed.index) # Evitar erro
        else:
            z_score_counts_for_hybrid = z_score_feature_counts_df[HYBRID_Z_SCORE_THRESHOLD]

        # Condição para ser outlier híbrido:
        # 1. Ser outlier pelo Isolation Forest (mask_if_from_best_results == True)
        # 2. Ter pelo menos MIN_Z_SCORE_FEATURES_COUNT features com Z-score > HYBRID_Z_SCORE_THRESHOLD
        hybrid_outliers_mask = mask_if_from_best_results & (z_score_counts_for_hybrid >= MIN_Z_SCORE_FEATURES_COUNT)
        # hybrid_outliers_mask está ALINHADA com df_processed

        num_hybrid_outliers = hybrid_outliers_mask.sum()
        perc_hybrid_outliers = (num_hybrid_outliers / len(df_processed)) * 100 if len(df_processed) > 0 else 0

        print(f"   Config Híbrida: Outlier pelo IF (melhor config) E Z-Score > {HYBRID_Z_SCORE_THRESHOLD} em >= {MIN_Z_SCORE_FEATURES_COUNT} feature(s).")
        print(f"   Outliers Detectados (Híbrido): {num_hybrid_outliers:,} ({perc_hybrid_outliers:.2f}%)")
    else:
        print("⚠️ Método Híbrido não aplicado: resultados do IF ou do Z-Score ausentes/desalinhados com df_processed.")

    # Comparação Final de Métodos (todos baseados em df_processed)
    if best_results and 'n_outliers' in best_results:
        print(f"\n🏆 COMPARAÇÃO FINAL DE MÉTODOS (todos sobre df_processed):")
        print(f"   • Isolation Forest (melhor config): {best_results.get('n_outliers',0):,} ({best_results.get('outlier_percentage',0):.2f}%)")

        if zscore_comparison_results:
            # Encontrar o Z-score threshold que resulta em % de outliers mais próxima do IF
            if best_results.get('outlier_percentage') is not None:
                 target_percentage_if = best_results['outlier_percentage']
                 closest_z_thresh = min(
                     zscore_comparison_results.keys(),
                     key=lambda z_thresh: abs(zscore_comparison_results[z_thresh]['percentage'] - target_percentage_if)
                 )
                 print(f"   • Z-Score (threshold {closest_z_thresh}, % mais próxima do IF): {zscore_comparison_results[closest_z_thresh]['total_outliers']:,} ({zscore_comparison_results[closest_z_thresh]['percentage']:.2f}%)")
            else:
                 print("   • Z-Score: % do IF não disponível para comparação de proximidade.")


        if 'num_hybrid_outliers' in locals() and len(hybrid_outliers_mask) > 0 : # Checar se o método híbrido foi calculado
            print(f"   • Método Híbrido (IF & Z-Score): {num_hybrid_outliers:,} ({perc_hybrid_outliers:.2f}%)")
else:
    print("❌ df_processed vazio ou sem final_numeric_cols. Comparação Z-Score/Híbrido não realizada.")


## Célula 9: Validação e Métricas de Qualidade (Simplificado)
# Esta célula é mais conceitual, pois "ground truth" de outliers geralmente não existe.
# Foca em consistência e separabilidade dos scores do IF.
validation_metrics_results = {}
overall_quality_score = 0.0

if best_results and 'outliers_mask' in best_results and 'scores' in best_results and not df_processed.empty and final_numeric_cols:

    def validate_outlier_detection_scores(
        df_val_proc,              # df_processed
        outliers_mask_val_proc,   # best_results['outliers_mask'] (do IF em df_processed)
        scores_val_proc,          # best_results['scores'] (do IF em df_processed)
        numeric_cols_val_proc     # final_numeric_cols
        ):
        print("\n✅ VALIDAÇÃO QUALITATIVA (Baseada no Isolation Forest Principal sobre df_processed)"); print("=" * 80)

        if len(outliers_mask_val_proc) == 0 or len(scores_val_proc) == 0 or df_val_proc.empty or not numeric_cols_val_proc:
            print("   Dados insuficientes para validação qualitativa.")
            return {}, 0.0
        if len(outliers_mask_val_proc) != len(df_val_proc) or len(scores_val_proc) != len(df_val_proc):
            print("   Desalinhamento entre máscaras/scores e df_processed. Validação pulada.")
            return {}, 0.0


        metrics = {}
        # 1. Qualidade de Separação dos Scores (reutilizando a função da classe IF)
        #    Esta métrica já foi calculada e está em best_results['separation_quality']
        metrics['separation_quality_if_scores'] = best_results.get('separation_quality', 0.0)
        print(f"   Métrica 1: Qualidade de Separação dos Scores (IF): {metrics['separation_quality_if_scores']:.4f}")
        print(f"     (Interpretação: Maior é melhor. Indica quão distintos são os scores de anomalia entre outliers e normais)")

        # 2. Consistência: Percentual de outliers
        #    Já está em best_results['outlier_percentage']
        metrics['percentage_outliers_if'] = best_results.get('outlier_percentage', 0.0)
        print(f"   Métrica 2: Percentual de Outliers Detectados (IF): {metrics['percentage_outliers_if']:.2f}%")
        print(f"     (Interpretação: Deve estar alinhado com o parâmetro 'contamination' esperado e o conhecimento do domínio)")

        # 3. Robustez: Estabilidade dos scores (usando desvio padrão dos scores)
        #    Um menor std pode indicar maior confiança nas magnitudes dos scores.
        #    Já está em best_results['score_std']
        metrics['score_std_if'] = best_results.get('score_std', 0.0)
        print(f"   Métrica 3: Desvio Padrão dos Scores de Anomalia (IF): {metrics['score_std_if']:.4f}")
        print(f"     (Interpretação: Relativo. Pode ser usado para comparar configurações. Menor pode ser preferível se a separação for boa)")


        # Cálculo de um "Overall Quality Score" simples e ponderado (exemplo)
        # Estes pesos são arbitrários e devem ser ajustados conforme a importância de cada métrica
        # Para separation_quality, assumimos que valores > 1 são bons. Normalizar para uma escala (ex: 0-1).
        # Um score de separação de 2 poderia ser mapeado para 1, 0 para 0. Limitar em um máximo razoável (ex: 5).
        norm_sep_qual = min(metrics['separation_quality_if_scores'] / 2.0, 1.0) if metrics['separation_quality_if_scores'] > 0 else 0.0

        # Para score_std, menor é melhor. Inverter e normalizar.
        # A normalização aqui é mais complexa e depende da escala esperada dos scores.
        # Simplificação: Se std for muito alto, penaliza.
        # Este é um placeholder, uma normalização mais robusta seria necessária.
        norm_score_std_inv = 1.0 / (1.0 + metrics['score_std_if']) if metrics['score_std_if'] > 0 else 0.5

        # Para percentage_outliers, quão próximo está do 'contamination' da melhor config.
        best_config_contamination = best_results.get('config', {}).get('contamination', 0.1) # default se não achar
        perc_diff_from_contamination = abs(metrics['percentage_outliers_if']/100 - best_config_contamination)
        # Queremos que a diferença seja pequena. 1 - diff.
        norm_perc_consistency = max(0, 1 - (perc_diff_from_contamination / best_config_contamination) if best_config_contamination >0 else 0)


        # Ponderação simples
        quality_score_val = (0.6 * norm_sep_qual) + \
                            (0.2 * norm_score_std_inv) + \
                            (0.2 * norm_perc_consistency)
        quality_score_val = max(0, min(quality_score_val, 1.0)) # Garantir que está entre 0 e 1

        print(f"   --------------------------------------------------------------------")
        print(f"   🎯 SCORE GERAL DE QUALIDADE DA DETECÇÃO (IF): {quality_score_val:.4f} (escala 0-1)")
        print(f"     (Este é um score composto e heurístico baseado nas métricas acima.)")
        return metrics, quality_score_val

    validation_metrics_results, overall_quality_score = validate_outlier_detection_scores(
        df_processed,
        best_results.get('outliers_mask', np.array([])),
        best_results.get('scores', np.array([])),
        final_numeric_cols
    )
else:
    print("⚠️ Validação qualitativa (Célula 9) pulada: dados insuficientes (best_results, df_processed, ou final_numeric_cols).")


## Célula 10: Análise Detalhada dos Outliers Detectados (no DataFrame Original 'df')
# O objetivo aqui é pegar as máscaras de df_processed (IF ou Híbrido) e tentar aplicá-las de volta ao df original.
# Isso requer que df_processed tenha mantido os índices originais de df.

if not df.empty: # Requer o DataFrame original
    print("\n🔍 ANÁLISE DETALHADA DOS OUTLIERS (aplicando máscaras ao DataFrame Original 'df')"); print("=" * 80)

    # Escolher qual máscara usar para a análise detalhada:
    # Prioridade: Híbrido (se calculado e alinhado com df_processed), senão IF (se calculado).
    analysis_mask_on_processed = np.array([]) # Esta será a máscara escolhida, ALINHADA COM df_processed
    analysis_method_name_detailed = "Nenhum"

    if 'hybrid_outliers_mask' in locals() and len(hybrid_outliers_mask) > 0 and len(hybrid_outliers_mask) == len(df_processed):
        analysis_mask_on_processed = hybrid_outliers_mask
        analysis_method_name_detailed = "Híbrido (IF + Z-Score)"
        print("   Usando máscara do Método Híbrido para análise detalhada.")
    elif best_results and 'outliers_mask' in best_results and len(best_results['outliers_mask']) == len(df_processed):
        analysis_mask_on_processed = best_results['outliers_mask']
        analysis_method_name_detailed = "Isolation Forest (melhor config)"
        print("   Usando máscara do Isolation Forest para análise detalhada (Híbrido não disponível/aplicável).")
    else:
        print("   Nenhuma máscara de outliers (IF ou Híbrido) utilizável de df_processed para análise detalhada. Pulando.")


    if len(analysis_mask_on_processed) > 0:
        # Agora, alinhar analysis_mask_on_processed (de df_processed) com o DataFrame original 'df'
        # Assumindo que df_processed.index é um subconjunto ou igual a df.index
        if df_processed.index.isin(df.index).all():
            analysis_mask_aligned_to_df_original = pd.Series(analysis_mask_on_processed, index=df_processed.index).reindex(df.index, fill_value=False).values
            print(f"   Máscara '{analysis_method_name_detailed}' alinhada com DataFrame original 'df' (Shape: {analysis_mask_aligned_to_df_original.shape}).")

            outliers_in_original_df = df.loc[analysis_mask_aligned_to_df_original].copy() # Usar .loc com a máscara booleana
            num_outliers_in_original = len(outliers_in_original_df)
            perc_outliers_in_original = (num_outliers_in_original / len(df)) * 100 if len(df) > 0 else 0

            print(f"   Total de outliers identificados no DataFrame original 'df' usando '{analysis_method_name_detailed}': {num_outliers_in_original:,} ({perc_outliers_in_original:.2f}%)")

            if num_outliers_in_original > 0:
                print(f"\n    primeiras 5 linhas dos outliers identificados em 'df':")
                display(outliers_in_original_df.head())

                print(f"\n   Estatísticas descritivas das COLUNAS NUMÉRICAS para os outliers em 'df':")
                if numeric_cols: # Usar as colunas numéricas originais
                    display(outliers_in_original_df[numeric_cols].describe(percentiles=[.1, .5, .9]))
                else:
                    print("      Nenhuma coluna numérica original para descrever.")

                # Comparar com as estatísticas dos dados normais no df original
                normals_in_original_df = df.loc[~analysis_mask_aligned_to_df_original]
                if not normals_in_original_df.empty and numeric_cols:
                    print(f"\n   Estatísticas descritivas das COLUNAS NUMÉRICAS para os dados NORMAIS em 'df':")
                    display(normals_in_original_df[numeric_cols].describe(percentiles=[.1, .5, .9]))


                # Análise de colunas categóricas (se houverem)
                if categorical_cols:
                    print(f"\n   Distribuição de valores em COLUNAS CATEGÓRICAS para os outliers em 'df':")
                    for cat_col in categorical_cols:
                        if cat_col in outliers_in_original_df.columns:
                            print(f"     Coluna '{cat_col}':")
                            display(outliers_in_original_df[cat_col].value_counts(normalize=True, dropna=False).head().to_frame())
            else:
                print("   Nenhum outlier encontrado no DataFrame original com a máscara selecionada.")
        else:
            print(f"   ⚠️ Não foi possível alinhar a máscara de '{analysis_method_name_detailed}' (de df_processed) com o DataFrame original 'df' devido a índices incompatíveis. Análise detalhada em 'df' pulada.")
    # else: já tratado acima se nenhuma máscara foi selecionada
else:
    print("❌ DataFrame original 'df' vazio. Análise detalhada (Célula 10) pulada.")


## Célula 11: Visualização de Outliers por Features
# Esta célula foi MODIFICADA para visualizar outliers em AMBOS df_processed e df original.

# Função auxiliar de visualização (pode ser definida uma vez e reutilizada)
def visualize_outliers_by_features(
    df_to_plot,                       # DataFrame para plotar (df ou df_processed)
    outlier_mask,                     # Máscara de outliers ALINHADA com df_to_plot
    title_suffix,                     # String para adicionar ao título (ex: "(Valores Processados)")
    method_name_for_title,            # Nome do método (ex: "Híbrido", "Isolation Forest")
    features_to_consider_for_plot,    # Lista de nomes de features a serem consideradas para plotagem
    n_features_to_plot=6
    ):

    actual_features_to_plot = [f for f in features_to_consider_for_plot if f in df_to_plot.columns][:n_features_to_plot]

    if not actual_features_to_plot:
        print(f"   ⚠️ Nenhuma feature selecionada ou disponível em '{title_suffix}' para visualização de outliers por features.")
        return

    print(f"   Visualizando outliers para as {len(actual_features_to_plot)} features em {title_suffix}: {actual_features_to_plot}")

    num_plots = len(actual_features_to_plot)
    cols_subplot = min(3, num_plots)
    rows_subplot = (num_plots + cols_subplot - 1) // cols_subplot

    fig_feat, axes_feat = plt.subplots(rows_subplot, cols_subplot, figsize=(18, 5 * rows_subplot), squeeze=False)
    fig_feat.suptitle(f'Distribuição de Outliers (Método: {method_name_for_title}) por Features {title_suffix}', fontsize=16, fontweight='bold')
    axes_feat = axes_feat.flatten()

    last_i_viz = 0 # Para rastrear o último índice de subplot usado
    for i_viz, feature_name in enumerate(actual_features_to_plot):
        if i_viz >= len(axes_feat): break
        last_i_viz = i_viz

        ax_current = axes_feat[i_viz]
        normal_data = df_to_plot.loc[df_to_plot.index[~outlier_mask], feature_name].dropna()
        outlier_data = df_to_plot.loc[df_to_plot.index[outlier_mask], feature_name].dropna()

        sns.histplot(normal_data, color="skyblue", label='Normal', kde=False, stat="density", element="step", ax=ax_current, bins=30, alpha=0.7)
        if not outlier_data.empty:
            sns.histplot(outlier_data, color="red", label='Outlier', kde=False, stat="density", element="step", ax=ax_current, bins=20, alpha=0.8)

        ax_current.set_title(f'{feature_name}', fontweight='bold', fontsize=12)
        ax_current.set_xlabel(f'Valor da Feature {title_suffix}', fontsize=10)
        ax_current.set_ylabel('Densidade', fontsize=10)
        ax_current.legend(fontsize=9)
        ax_current.grid(True, linestyle='--', alpha=0.5)

        legend_handles_means = []
        if not normal_data.empty:
            mean_normal = normal_data.mean()
            ax_current.axvline(mean_normal, color='blue', linestyle='--', linewidth=1.5, alpha=0.9)
            legend_handles_means.append(plt.Line2D([0], [0], color='blue', linestyle='--', label=f'Média Normal: {mean_normal:.2f}'))
        if not outlier_data.empty:
            mean_outlier = outlier_data.mean()
            ax_current.axvline(mean_outlier, color='darkred', linestyle=':', linewidth=1.5, alpha=0.9)
            legend_handles_means.append(plt.Line2D([0], [0], color='darkred', linestyle=':', label=f'Média Outlier: {mean_outlier:.2f}'))

        if legend_handles_means:
            current_handles, current_labels = ax_current.get_legend_handles_labels()
            unique_handles = []
            unique_labels = []
            temp_legend_items = list(zip(current_handles + legend_handles_means, current_labels + [h.get_label() for h in legend_handles_means]))
            for handle, label in temp_legend_items:
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax_current.legend(handles=unique_handles, labels=unique_labels, fontsize=9)

    for j_viz in range(last_i_viz + 1, len(axes_feat)):
        if j_viz < len(axes_feat):
             fig_feat.delaxes(axes_feat[j_viz])

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.show()

# --- Início da Lógica da Célula 11 ---
print("\n🖼️ CÉLULA 11: VISUALIZAÇÃO DE OUTLIERS POR FEATURES")
print("=" * 60)

# --- Parte 1: Visualização sobre df_processed ---
if 'df_processed' in globals() and not df_processed.empty and \
   'best_results' in globals() and best_results and 'outliers_mask' in best_results and \
   'final_numeric_cols' in globals() and final_numeric_cols:

    print("\n--- Visualização sobre o DataFrame PROCESSADO (df_processed) ---")

    mask_for_processed_viz = np.array([])
    method_name_processed = "Nenhum"

    # Escolher máscara para df_processed (Híbrido ou IF)
    if 'hybrid_outliers_mask' in locals() and len(hybrid_outliers_mask) > 0 and len(hybrid_outliers_mask) == len(df_processed):
        mask_for_processed_viz = hybrid_outliers_mask
        method_name_processed = "Híbrido (IF + Z-Score)"
    elif 'outliers_mask' in best_results and len(best_results['outliers_mask']) == len(df_processed):
        mask_for_processed_viz = best_results['outliers_mask']
        method_name_processed = "Isolation Forest"

    if len(mask_for_processed_viz) > 0 :
        # Features para visualização em df_processed: usar final_numeric_cols ou feature_importance
        features_list_for_processed = []
        if 'feature_importance' in globals() and feature_importance:
            features_list_for_processed = [f for f in list(feature_importance.keys()) if f in final_numeric_cols] # Garantir que as features de importância estão nas processadas
            if not features_list_for_processed: # Fallback se nenhuma feature de importância estiver em final_numeric_cols
                 features_list_for_processed = final_numeric_cols
        elif final_numeric_cols:
            features_list_for_processed = final_numeric_cols

        if features_list_for_processed:
            visualize_outliers_by_features(
                df_processed,
                mask_for_processed_viz,
                "(Valores Processados)",
                method_name_processed,
                features_list_for_processed,
                n_features_to_plot=6
            )
        else:
            print("   ⚠️ Nenhuma feature apropriada (final_numeric_cols ou feature_importance) para visualização em df_processed.")
    else:
        print("   ⚠️ Nenhuma máscara de outlier aplicável encontrada para df_processed. Visualização pulada.")
else:
    print("\n⚠️ Visualização sobre df_processed pulada: df_processed, best_results ou final_numeric_cols não disponíveis/completos.")


# --- Parte 2: Visualização sobre df ORIGINAL ---
if 'df' in globals() and not df.empty and \
   'df_processed' in globals() and not df_processed.empty and \
   'best_results' in globals() and best_results and 'outliers_mask' in best_results:

    print("\n--- Visualização sobre o DataFrame ORIGINAL (df) ---")

    mask_source_on_processed_for_original_viz = np.array([])
    method_name_original = "Nenhum"

    if 'hybrid_outliers_mask' in locals() and len(hybrid_outliers_mask) > 0 and len(hybrid_outliers_mask) == len(df_processed):
        mask_source_on_processed_for_original_viz = hybrid_outliers_mask
        method_name_original = "Híbrido (IF + Z-Score)"
    elif 'outliers_mask' in best_results and len(best_results['outliers_mask']) == len(df_processed):
        mask_source_on_processed_for_original_viz = best_results['outliers_mask']
        method_name_original = "Isolation Forest"

    display_mask_aligned_to_df_original = np.array([])
    if len(mask_source_on_processed_for_original_viz) > 0:
        if df_processed.index.isin(df.index).all() and df.index.isin(df_processed.index).all() and len(df.index.unique()) == len(df_processed.index.unique()): # Checagem mais robusta de alinhamento de índice
            try:
                display_mask_aligned_to_df_original = pd.Series(mask_source_on_processed_for_original_viz, index=df_processed.index).reindex(df.index, fill_value=False).values
                if len(display_mask_aligned_to_df_original) != len(df): # Double check length after reindex
                     print(f"   ⚠️ Desalinhamento de comprimento após reindexação para df original. Máscara: {len(display_mask_aligned_to_df_original)}, df: {len(df)}. Pulando visualização em df.")
                     display_mask_aligned_to_df_original = np.array([]) # Resetar para pular
                else:
                     print(f"   Máscara do método '{method_name_original}' (de df_processed) foi alinhada com 'df' para visualização.")
            except Exception as e_reindex:
                 print(f"   ⚠️ Erro ao reindexar máscara para df original: {e_reindex}. Pulando visualização em df.")
                 display_mask_aligned_to_df_original = np.array([])
        else:
            print(f"   ⚠️ Não foi possível alinhar a máscara do método '{method_name_original}' com 'df' (índices não totalmente compatíveis). Visualização em 'df' pulada.")

    if len(display_mask_aligned_to_df_original) > 0 and len(display_mask_aligned_to_df_original) == len(df):
        features_list_for_original_df = []
        if 'feature_importance' in globals() and feature_importance:
            features_list_for_original_df = [f for f in list(feature_importance.keys()) if f in df.columns] # Features de importância que existem em df
            if not features_list_for_original_df and 'numeric_cols' in globals() and numeric_cols: # Fallback se nenhuma feature de importância estiver em df
                features_list_for_original_df = [f for f in numeric_cols if f in df.columns]
        elif 'numeric_cols' in globals() and numeric_cols:
            features_list_for_original_df = [f for f in numeric_cols if f in df.columns]

        if features_list_for_original_df:
            visualize_outliers_by_features(
                df,
                display_mask_aligned_to_df_original,
                "(Valores Originais de 'df')",
                method_name_original,
                features_list_for_original_df,
                n_features_to_plot=6
            )
        else:
             print("   ⚠️ Nenhuma feature apropriada (colunas numéricas originais ou de importância) para visualização em df original.")
    elif len(mask_source_on_processed_for_original_viz) > 0 : # Se havia uma máscara fonte mas o alinhamento falhou
        print(f"   ⚠️ Visualização em 'df' pulada devido a falha no alinhamento da máscara '{method_name_original}'.")
else:
    print("\n⚠️ Visualização de outliers por features (Célula 11) completamente pulada: condições iniciais não atendidas (df, df_processed ou best_results).")


## Célula 12_A: Exportação de Datasets Processados e Tratamento Geral de Outliers
print("\n💾 CÉLULA 12_A: EXPORTAÇÃO DE DATASETS PROCESSADOS E TRATADOS"); print("=" * 60)
df_processed_winsorized = pd.DataFrame()      # df_processed após winsorização
df_processed_outliers_removed = pd.DataFrame() # df_processed com outliers (IF ou Híbrido) removidos

if 'df_processed' in globals() and not df_processed.empty:
    try:
        df_processed.to_csv('dataset_somente_preprocessado.csv', index=False)
        print("✅ 'dataset_somente_preprocessado.csv' salvo (escalonado, limpo, baseado em df_processed).")

        if 'final_numeric_cols' in globals() and final_numeric_cols:
            # Winsorização em df_processed
            df_processed_winsorized = df_processed.copy()
            print("\n🔧 Aplicando Winsorização (5%-95%) às colunas numéricas de 'df_processed'...")
            for col_win in final_numeric_cols: # Usar final_numeric_cols que são as efetivamente processadas
                if col_win in df_processed_winsorized.columns:
                    # Calcular quantis sobre os dados não-NaN da coluna
                    valid_data_win = df_processed_winsorized[col_win].dropna()
                    if not valid_data_win.empty:
                        low_quantile, high_quantile = valid_data_win.quantile([0.05, 0.95])
                        df_processed_winsorized[col_win] = df_processed_winsorized[col_win].clip(low_quantile, high_quantile)
                    else:
                        print(f"   Aviso Winsorização: Coluna '{col_win}' está vazia após dropna, não foi winsorizada.")

            df_processed_winsorized.to_csv('dataset_preprocessado_tratado_winsorizado.csv', index=False)
            print("✅ 'dataset_preprocessado_tratado_winsorizado.csv' salvo (df_processed + winsorização).")


            # Remoção de outliers de df_processed
            print("\n🔧 Removendo outliers identificados de 'df_processed'...")
            # Usar a máscara do método híbrido se disponível e alinhada com df_processed, senão IF
            mask_for_removal_on_processed = np.array([])
            method_for_removal_name = "Nenhum"

            if 'hybrid_outliers_mask' in locals() and len(hybrid_outliers_mask) > 0 and len(hybrid_outliers_mask) == len(df_processed):
                mask_for_removal_on_processed = hybrid_outliers_mask
                method_for_removal_name = "Híbrido (IF + Z-Score)"
            elif 'best_results' in globals() and best_results and 'outliers_mask' in best_results and len(best_results['outliers_mask']) == len(df_processed):
                mask_for_removal_on_processed = best_results['outliers_mask']
                method_for_removal_name = "Isolation Forest (melhor config)"

            if len(mask_for_removal_on_processed) > 0:
                # `~mask_for_removal_on_processed` seleciona os NÃO outliers (dados limpos)
                df_processed_outliers_removed = df_processed[~mask_for_removal_on_processed].copy()
                df_processed_outliers_removed.to_csv('dataset_preprocessado_sem_outliers_identificados.csv', index=False)
                print(f"✅ 'dataset_preprocessado_sem_outliers_identificados.csv' salvo.")
                print(f"   (Outliers removidos de 'df_processed' usando o método: {method_for_removal_name}).")
                print(f"   Shape original df_processed: {df_processed.shape}, Shape após remoção de outliers: {df_processed_outliers_removed.shape}")
            else:
                print("   ⚠️ Não foi possível determinar máscara para remoção de outliers de 'df_processed' ou máscara não alinhada. Remoção pulada.")
        else:
            print("⚠️ Winsorização/Remoção de Outliers de df_processed pulada: 'final_numeric_cols' não encontrado ou vazio.")
    except Exception as e_exp_proc:
        print(f"❌ Erro durante exportação/tratamento na Célula 12_A: {e_exp_proc}")
else:
    print("⚠️ 'df_processed' não está definido ou está vazio. Exportações/Tratamentos da Célula 12_A pulados.")


## Célula 13: Relatório Final e Recomendações
final_report_data_dict = {} # Dicionário para armazenar os dados do relatório

# Checar se as variáveis necessárias para o relatório existem
report_possible = (
    'best_results' in globals() and best_results and 'config' in best_results and
    not df.empty and
    'df_processed' in globals() # df_processed é usado para % híbrida
)

if report_possible:
    def generate_final_report(
        df_original_rep,                # df
        df_processed_rep,               # df_processed
        best_results_if_rep,            # best_results (do IF em df_processed)
        hybrid_mask_on_processed_rep,   # hybrid_outliers_mask (em df_processed)
        validation_metrics_if_rep,      # validation_metrics_results
        quality_score_if_rep,           # overall_quality_score (do IF)
        feat_importance_if_rep=None,    # feature_importance (do IF)
        final_numeric_cols_rep=None     # final_numeric_cols
        ):
        print("\n📋 RELATÓRIO FINAL - DETECÇÃO DE OUTLIERS"); print("=" * 41)

        report_obj = {} # Para coletar dados para retorno

        print(f"🗂️  INFORMAÇÕES DO DATASET ORIGINAL ('df'):")
        print(f"   • Total de registros (linhas): {len(df_original_rep):,}")
        report_obj['total_records_original'] = len(df_original_rep)
        if 'numeric_cols' in globals() and numeric_cols:
            print(f"   • Colunas numéricas originais: {len(numeric_cols)}")
            report_obj['num_numeric_cols_original'] = len(numeric_cols)
        if 'categorical_cols' in globals() and categorical_cols:
            print(f"   • Colunas categóricas originais: {len(categorical_cols)}")
            report_obj['num_categorical_cols_original'] = len(categorical_cols)

        print(f"\n⚙️  PRÉ-PROCESSAMENTO (gerando 'df_processed'):")
        print(f"   • Shape de 'df_processed': {df_processed_rep.shape if not df_processed_rep.empty else 'N/A'}")
        report_obj['df_processed_shape'] = df_processed_rep.shape if not df_processed_rep.empty else None
        if scaler_used:
             print(f"   • Scaler utilizado: {type(scaler_used).__name__}")
             report_obj['scaler_used'] = type(scaler_used).__name__
        if final_numeric_cols_rep:
            print(f"   • Colunas numéricas em 'df_processed' (final_numeric_cols): {len(final_numeric_cols_rep)}")
            report_obj['num_final_numeric_cols_in_processed'] = len(final_numeric_cols_rep)


        print(f"\n🎯 RESULTADOS DA DETECÇÃO (Isolation Forest Principal - sobre 'df_processed'):")
        if best_results_if_rep:
            print(f"   • Melhor Configuração IF: {best_results_if_rep.get('config', 'N/A')}")
            print(f"   • Outliers detectados (IF): {best_results_if_rep.get('n_outliers',0):,} ({best_results_if_rep.get('outlier_percentage',0):.2f}%)")
            print(f"   • Qualidade de Separação dos Scores (IF): {best_results_if_rep.get('separation_quality',0):.4f}")
            report_obj['if_best_config'] = best_results_if_rep.get('config')
            report_obj['if_n_outliers_on_processed'] = best_results_if_rep.get('n_outliers',0)
            report_obj['if_percentage_on_processed'] = best_results_if_rep.get('outlier_percentage',0)
            report_obj['if_separation_quality'] = best_results_if_rep.get('separation_quality',0)


        print(f"\n🧬 RESULTADOS DA DETECÇÃO (Método Híbrido - sobre 'df_processed'):")
        if len(hybrid_mask_on_processed_rep) > 0 and len(hybrid_mask_on_processed_rep) == len(df_processed_rep):
            num_hybrid_outliers_rep = hybrid_mask_on_processed_rep.sum()
            perc_hybrid_outliers_rep = (num_hybrid_outliers_rep / len(df_processed_rep)) * 100 if len(df_processed_rep) > 0 else 0
            print(f"   • Outliers detectados (Híbrido): {num_hybrid_outliers_rep:,} ({perc_hybrid_outliers_rep:.2f}%)")
            report_obj['hybrid_n_outliers_on_processed'] = num_hybrid_outliers_rep
            report_obj['hybrid_percentage_on_processed'] = perc_hybrid_outliers_rep
        else:
            print(f"   • Resultados do Método Híbrido não disponíveis ou máscara não alinhada com 'df_processed'.")
            report_obj['hybrid_n_outliers_on_processed'] = None
            report_obj['hybrid_percentage_on_processed'] = None


        print(f"\n🏅 MÉTRICAS DE QUALIDADE (Isolation Forest Principal):")
        if validation_metrics_if_rep:
            for k_met, v_met in validation_metrics_if_rep.items(): print(f"   • {k_met.replace('_', ' ').capitalize()}: {v_met:.4f}")
        print(f"   • Score Geral de Qualidade (IF): {quality_score_if_rep:.4f}")
        report_obj['if_validation_metrics'] = validation_metrics_if_rep
        report_obj['if_overall_quality_score'] = quality_score_if_rep


        if feat_importance_if_rep:
            print(f"\n✨ FEATURES MAIS IMPORTANTES (pelo IF em 'df_processed' - Top 5):")
            top_5_features = list(feat_importance_if_rep.items())[:5]
            for feat_name_imp, imp_val in top_5_features: print(f"   • {feat_name_imp}: {imp_val:.4f}")
            report_obj['if_top_features'] = dict(top_5_features)


        print(f"\n💡 RECOMENDAÇÕES E PRÓXIMOS PASSOS:")
        print(f"   • Analisar os arquivos CSV exportados na Célula 12_A e Célula 14:")
        print(f"     - 'dataset_somente_preprocessado.csv': Dados limpos e escalonados (base 'df_processed').")
        print(f"     - 'dataset_preprocessado_tratado_winsorizado.csv': 'df_processed' com winsorização aplicada.")
        print(f"     - 'dataset_preprocessado_sem_outliers_identificados.csv': 'df_processed' com outliers (IF ou Híbrido) removidos.")
        print(f"     - 'dataset_original_com_flags_outliers.csv': DataFrame original 'df' com colunas adicionais indicando scores e flags de outlier (IF e Híbrido).")
        print(f"     - 'outliers_detectados_if_do_original.csv': Apenas as linhas de 'df' que foram flagadas como outliers pelo IF (após alinhamento).")
        print(f"   • Revisar as visualizações geradas (Célula 7 e Célula 11) para entender a natureza dos outliers.")
        print(f"   • Se os outliers representarem erros de dados, considere corrigi-los ou removê-los. Se forem anomalias genuínas, investigue suas causas.")
        print(f"   • O 'Score Geral de Qualidade' é uma métrica heurística. Use-o em conjunto com o conhecimento do domínio para avaliar os resultados.")
        print(f"   • Experimentar com diferentes configurações de 'contamination' no Isolation Forest e parâmetros do Z-Score pode refinar os resultados.")

        return report_obj

    final_report_data_dict = generate_final_report(
        df,
        df_processed,
        best_results, # Resultados do IF em df_processed
        hybrid_outliers_mask if 'hybrid_outliers_mask' in locals() else np.array([]), # Máscara híbrida em df_processed
        validation_metrics_results, # Métricas do IF
        overall_quality_score,      # Score de qualidade do IF
        feature_importance,
        final_numeric_cols
    )
else:
    print("⚠️ Relatório final (Célula 13) não gerado: dados ou resultados insuficientes.")
    print("   Verifique se 'df', 'df_processed' e 'best_results' (do IF) estão disponíveis e corretos.")


## Célula 14: Exportação dos Resultados Finais (DataFrame Original 'df' + Flags)
# Esta célula adiciona colunas ao DataFrame ORIGINAL 'df' com os scores e flags de outlier.
# Requer que as máscaras e scores (originalmente de df_processed) possam ser alinhadas com 'df'.

export_success_cell14 = False
if not df.empty and 'df_processed' in globals() and not df_processed.empty and 'best_results' in globals() and best_results:
    print("\n💾 CÉLULA 14: EXPORTAÇÃO DO DATASET ORIGINAL 'df' COM FLAGS DE OUTLIER"); print("=" * 75)

    df_original_with_flags = df.copy() # Começar com uma cópia do df original

    # 1. Alinhar Scores do Isolation Forest (de df_processed para df)
    #    best_results['scores'] está alinhado com df_processed.index
    if 'scores' in best_results and len(best_results['scores']) == len(df_processed):
        scores_if_on_processed = pd.Series(best_results['scores'], index=df_processed.index)
        # Reindexar para o índice de df. fill_value pode ser np.nan ou outro indicador.
        df_original_with_flags['anomaly_score_if'] = scores_if_on_processed.reindex(df_original_with_flags.index, fill_value=np.nan)
        # Rank dos scores (menor score = mais anômalo, então rank ascendente)
        df_original_with_flags['outlier_rank_if'] = df_original_with_flags['anomaly_score_if'].rank(method='dense', ascending=True).astype('Int64') # Usar Int64 para permitir NaNs
        print("   ✅ Colunas 'anomaly_score_if' e 'outlier_rank_if' adicionadas a 'df_original_with_flags'.")
    else:
        print("   ⚠️ Scores do IF não disponíveis ou desalinhados com df_processed. Colunas de score IF não adicionadas a 'df_original_with_flags'.")
        df_original_with_flags['anomaly_score_if'] = np.nan # Adicionar coluna com NaNs
        df_original_with_flags['outlier_rank_if'] = pd.NA


    # 2. Alinhar Máscara do Isolation Forest (de df_processed para df)
    #    best_results['outliers_mask'] está alinhado com df_processed.index
    if 'outliers_mask' in best_results and len(best_results['outliers_mask']) == len(df_processed):
        mask_if_on_processed = pd.Series(best_results['outliers_mask'], index=df_processed.index)
        df_original_with_flags['is_outlier_if'] = mask_if_on_processed.reindex(df_original_with_flags.index, fill_value=False).astype(bool) # False para não encontrados
        print("   ✅ Coluna 'is_outlier_if' adicionada a 'df_original_with_flags'.")
    else:
        print("   ⚠️ Máscara do IF não disponível ou desalinhada. Coluna 'is_outlier_if' não adicionada/preenchida com False.")
        df_original_with_flags['is_outlier_if'] = False


    # 3. Alinhar Máscara Híbrida (de df_processed para df)
    #    hybrid_outliers_mask está alinhado com df_processed.index
    if 'hybrid_outliers_mask' in locals() and len(hybrid_outliers_mask) > 0 and len(hybrid_outliers_mask) == len(df_processed):
        mask_hybrid_on_processed = pd.Series(hybrid_outliers_mask, index=df_processed.index)
        df_original_with_flags['is_outlier_hybrid'] = mask_hybrid_on_processed.reindex(df_original_with_flags.index, fill_value=False).astype(bool)
        print("   ✅ Coluna 'is_outlier_hybrid' adicionada a 'df_original_with_flags'.")
    else:
        print("   ⚠️ Máscara Híbrida não disponível ou desalinhada. Coluna 'is_outlier_hybrid' não adicionada/preenchida com False.")
        df_original_with_flags['is_outlier_hybrid'] = False


    try:
        df_original_with_flags.to_csv('dataset_original_com_flags_outliers.csv', index=False)
        print("\n   ➡️ 'dataset_original_com_flags_outliers.csv' EXPORTADO COM SUCESSO.")
        export_success_cell14 = True

        # Exportar apenas os outliers identificados pelo IF no DataFrame original
        if 'is_outlier_if' in df_original_with_flags.columns:
            outliers_df_if_from_original = df_original_with_flags[df_original_with_flags['is_outlier_if'] == True].copy()
            if not outliers_df_if_from_original.empty:
                outliers_df_if_from_original.to_csv('outliers_detectados_if_do_original.csv', index=False)
                print(f"   ➡️ 'outliers_detectados_if_do_original.csv' EXPORTADO ({len(outliers_df_if_from_original)} linhas).")
            else:
                print("   ℹ️ Nenhum outlier identificado pelo IF no DataFrame original para exportar em 'outliers_detectados_if_do_original.csv'.")
    except Exception as e_export_final:
        print(f"   ❌ ERRO ao exportar arquivos na Célula 14: {e_export_final}")

else:
    print("⚠️ Exportação da Célula 14 pulada: DataFrame original 'df', 'df_processed' ou resultados do IF ('best_results') ausentes/incompletos.")

if export_success_cell14:
    print("\n🎉 ANÁLISE E EXPORTAÇÃO DA BASE ORIGINAL COM FLAGS CONCLUÍDA!")
else:
    print("\nℹ️ Exportação principal da Célula 14 não foi concluída ou foi pulada.")


print("\n" + "="*50)
print("🏁 FIM DA ANÁLISE DE DETECÇÃO DE OUTLIERS")
print("="*50)



# %%
