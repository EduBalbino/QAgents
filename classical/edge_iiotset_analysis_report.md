# Relatório de Análise Estatística do Dataset Edge-IIoTset
*Gerado em: 11-02-2026 18:28:41*

## 1. Introdução

Este relatório apresenta uma análise estatística detalhada do dataset Edge-IIoTset, que contém dados relacionados à segurança cibernética em ambientes de IoT (Internet das Coisas) e IIoT (Internet das Coisas Industrial). A análise foi conduzida utilizando Python com as bibliotecas pandas, numpy, matplotlib e seaborn.


## 2. Aquisição e Preparação dos Dados

### 2.1 Fonte dos Dados
O dataset foi obtido do Kaggle, especificamente do conjunto "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot".

### 2.2 Processo de Extração
- Os arquivos foram carregados a partir do diretório local `data/CIC-BCCC-*`
- Foi realizada uma verificação recursiva de arquivos CSV em 22 arquivos, tanto em arquivos ZIP quanto em diretórios
- Os arquivos CSV encontrados foram:
  - Backdoor.csv
  - DDoS HTTP Flood.csv
  - Uploading Attack.csv
  - Ransomware.csv
  - Port Scanning.csv
  - ... e mais 17 arquivos


## 3. Características Gerais do Dataset

### 3.1 Dimensões
- **Total de Registros**: 900
- **Total de Features**: 85

### 3.2 Tipos de Variáveis
- **Variáveis Numéricas**: 80
- **Variáveis Categóricas**: 5
- **Variáveis Binárias**: 10

#### Exemplo de Variáveis por Tipo:
**Numéricas**: tcp.srcport, tcp.dstport, protocol, flow.duration, total.fwd.packet e mais 75

**Categóricas**: flow.id, ip.src_host, ip.dst_host, frame.time, Attack_label

**Binárias**: total.length.of.bwd.packet, bwd.packet.length.max, bwd.packet.length.min, bwd.packet.length.mean, fwd.psh.flags e mais 5


## 4. Análise de Classes/Ataques
### 4.1 Distribuição das Classes

#### Classe: Attack_label
| Valor | Contagem | Percentual |
|-------|----------|------------|
| Backdoor | 900 | 100.00% |

![Distribuição da Classe Attack_label](images/class_distribution_Attack_label.png)

#### Classe: Attack_type
| Valor | Contagem | Percentual |
|-------|----------|------------|
| 1 | 900 | 100.00% |

![Distribuição da Classe Attack_type](images/class_distribution_Attack_type.png)


## 5. Análise Estatística das Features Numéricas

### 5.1 Estatísticas Descritivas
| Feature | Contagem | Média | Desvio Padrão | Mínimo | 25% | 50% (Mediana) | 75% | Máximo |
|---------|----------|-------|---------------|--------|-----|--------------|-----|--------|
| tcp.srcport | 900 | 53,026.28 | 13,384.76 | 4,321.00 | 58,525.50 | 58,975.00 | 60,548.50 | 60,998.00 |
| tcp.dstport | 900 | 5,716.39 | 8,436.78 | 4,321.00 | 4,321.00 | 4,321.00 | 4,321.00 | 60,212.00 |
| protocol | 900 | 6.00 | 0.00 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| flow.duration | 900 | 7,746,269.84 | 27,878,464.63 | 74.00 | 3,443.75 | 3,800.00 | 6,092.25 | 119,992,365.00 |
| total.fwd.packet | 900 | 13.61 | 53.01 | 1.00 | 1.00 | 1.00 | 1.00 | 249.00 |
| total.bwd.packets | 900 | 13.09 | 51.07 | 0.00 | 1.00 | 1.00 | 1.00 | 236.00 |
| total.length.of.fwd.packet | 900 | 5,142.51 | 21,747.05 | 0.00 | 0.00 | 0.00 | 0.00 | 108,528.00 |
| total.length.of.bwd.packet | 900 | 6.72 | 43.50 | 0.00 | 0.00 | 0.00 | 0.00 | 288.00 |
| fwd.packet.length.max | 900 | 80.96 | 325.80 | 0.00 | 0.00 | 0.00 | 0.00 | 1,448.00 |
| fwd.packet.length.min | 900 | 5.97 | 28.71 | 0.00 | 0.00 | 0.00 | 0.00 | 208.00 |

*Nota: Apenas as 10 primeiras features numéricas são exibidas de um total de 80.*

### 5.2 Correlação entre Features

![Matriz de Correlação](images/correlation_heatmap.png)

#### Correlações Significativas Identificadas:
| Feature 1 | Feature 2 | Correlação |
|-----------|-----------|------------|
| total.length.of.bwd.packet | bwd.packet.length.max | 1.0000 |
| total.length.of.bwd.packet | bwd.packet.length.min | 1.0000 |
| total.length.of.bwd.packet | bwd.packet.length.mean | 1.0000 |
| bwd.packet.length.max | bwd.packet.length.min | 1.0000 |
| bwd.packet.length.max | bwd.packet.length.mean | 1.0000 |
| bwd.packet.length.min | bwd.packet.length.mean | 1.0000 |
| total.fwd.packet | total.bwd.packets | 0.9998 |
| total.fwd.packet | total.length.of.fwd.packet | 0.9972 |
| total.bwd.packets | total.length.of.fwd.packet | 0.9959 |
| fwd.packet.length.max | fwd.packet.length.mean | 0.9926 |

*Nota: Exibindo apenas as 10 correlações mais fortes de um total de 38 correlações significativas.*


## 6. Análise de Outliers

### Feature: tcp.srcport
- **Outliers detectados**: 201 (22.33% dos registros)
- **Limite inferior**: 55491.00
- **Limite superior**: 63583.00
- **Valor mínimo de outlier**: 4321.00
- **Valor máximo de outlier**: 33120.00

![Boxplot para tcp.srcport](images/boxplot_tcp.srcport.png)

### Feature: tcp.dstport
- **Outliers detectados**: 24 (2.67% dos registros)
- **Limite inferior**: 4321.00
- **Limite superior**: 4321.00
- **Valor mínimo de outlier**: 56324.00
- **Valor máximo de outlier**: 60212.00

![Boxplot para tcp.dstport](images/boxplot_tcp.dstport.png)

### Feature: protocol
- **Outliers detectados**: 0 (0.00% dos registros)
- **Limite inferior**: 6.00
- **Limite superior**: 6.00

![Boxplot para protocol](images/boxplot_protocol.png)

### Feature: flow.duration
- **Outliers detectados**: 174 (19.33% dos registros)
- **Limite inferior**: -529.00
- **Limite superior**: 10065.00
- **Valor mínimo de outlier**: 10078.00
- **Valor máximo de outlier**: 119992365.00

![Boxplot para flow.duration](images/boxplot_flow.duration.png)

### Feature: total.fwd.packet
- **Outliers detectados**: 73 (8.11% dos registros)
- **Limite inferior**: 1.00
- **Limite superior**: 1.00
- **Valor mínimo de outlier**: 2.00
- **Valor máximo de outlier**: 249.00

![Boxplot para total.fwd.packet](images/boxplot_total.fwd.packet.png)


## 7. Análise de Valores Ausentes

Não foram encontrados valores ausentes no dataset analisado.


## 8. Conclusões e Próximos Passos

### 8.1 Principais Conclusões
- O dataset Edge-IIoTset contém dados relacionados à segurança cibernética em ambientes IoT e IIoT
- Foram analisados 900 registros com 85 features
- Foram identificadas 2 colunas de classe/tipo de ataque
- Presença significativa de outliers nas features: tcp.srcport, flow.duration, total.fwd.packet
- Foram identificadas 38 correlações fortes entre features
- O dataset não apresenta valores ausentes

### 8.2 Recomendações para Análises Futuras

- **Pré-processamento de Dados**:
  - Tratar valores ausentes conforme recomendado na seção 7
  - Normalizar ou padronizar features numéricas para melhorar desempenho de algoritmos
  - Tratar outliers através de técnicas como winsorização ou transformações logarítmicas

- **Engenharia de Features**:
  - Investigar possíveis combinações ou transformações de features existentes
  - Reduzir dimensionalidade através de PCA ou seleção de features para melhorar desempenho

- **Modelagem**:

- **Avaliação e Interpretação**:
  - Aplicar validação cruzada para avaliar robustez dos modelos
  - Analisar importância de features para identificar quais são mais relevantes na detecção de ataques
  - Implementar explicabilidade (SHAP, LIME) para entender decisões dos modelos

- **Implementação em Ambientes Reais**:
  - Estabelecer pipeline de monitoramento para detectar drift nos dados
  - Avaliar desempenho computacional para implementação em dispositivos edge
