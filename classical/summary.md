# Resumo do Trabalho (Base IoT + PLS)

\section{Base de dados}
A base de dados utilizada neste trabalho foi construída a partir de um recorte IoT do ecossistema **CIC-BCCC-NRC TabularIoTAttack-2024**, com foco em três coleções centrais: `CIC-BCCC-NRC-Edge-IIoTSet-2022`, `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` e `CIC-BCCC-NRC-UQ-IOT-2022`. Em termos práticos, o objetivo foi consolidar tráfego de rede IoT/IIoT com comportamento benigno e malicioso em um único cenário experimental, mantendo rastreabilidade da origem das classes e padronização de nomenclaturas para uso em aprendizado de máquina.

O processo de integração resultou em um artefato único (`data/processed/trio_multiclass_final_single.csv`), com colunas normalizadas no padrão `lowercase.dot` e duas variáveis-alvo principais: `Attack_label` (multiclasse) e `Attack_type` (indicador de ataque). Na etapa de análise estatística exploratória, foi trabalhado um recorte com **200.000 registros** e **85 features**, incluindo variáveis numéricas e categóricas de fluxo, com distribuição de classes fortemente desbalanceada em `Attack_label` e desbalanceamento moderado em `Attack_type`.

Para organização semântica das classes, foi adotada uma taxonomia final de ataques especialistas (Backdoor, DDoS\_HTTP, DDoS\_ICMP, DDoS\_TCP, DDoS\_UDP, Fingerprinting, MITM, Password, Port\_Scanning, Ransomware, SQL\_injection, Uploading, Vulnerability\_scanner, XSS), além de `Others` para eventos não mapeados diretamente, e `NORMAL` para tráfego benigno.

\subsection{Pré-processamento da base de dados}
Em uma primeira etapa, o conjunto passou por inspeção estatística descritiva, análise de distribuição de classes e verificação de consistência dos tipos. Nessa fase, os dados foram preparados para reduzir ruído de medição e melhorar a estabilidade das etapas supervisionadas, preservando o cenário realista de tráfego IoT/IIoT. A análise também evidenciou correlações fortes entre pares de atributos de comprimento de pacotes e presença de outliers relevantes em variáveis como `tcp.dstport` e `flow.duration`, o que reforçou a necessidade de um pipeline robusto de transformação.

No recorte analisado, não foram identificados valores ausentes. Ainda assim, manteve-se uma estratégia conservadora de preparação para evitar vazamento entre treino e teste, com todas as transformações aprendidas no treino e apenas aplicadas no teste.

\subsubsection{Redução supervisionada com PLS}
Após a preparação inicial, foi aplicada **Partial Least Squares (PLS)** como etapa de redução supervisionada de dimensionalidade, com o objetivo de projetar o espaço original de atributos em componentes mais informativos para discriminar tráfego benigno e malicioso. Diferentemente de reduções puramente não supervisionadas, o PLS foi escolhido por incorporar a informação do alvo durante a projeção, favorecendo componentes com maior relevância preditiva para a tarefa de detecção.

No pipeline adotado, o ajuste foi feito somente com o conjunto de treino (sem leakage), usando `PLSRegression(n_components=8)`. Em seguida, os componentes gerados (`PC_1` a `PC_8`) foram transformados por mapeamento quantílico para distribuição uniforme em \([0,1]\), também ajustado apenas no treino e reaplicado no teste. O resultado final foi um dataset derivado (`data/processed/mergido_preprocessado.csv`) com representação compacta e estável para os experimentos subsequentes.

\subsubsection{Detecção de outliers}
A identificação de anomalias foi tratada como parte da higienização da base para reduzir efeitos de caudas extremas antes da modelagem. No estudo exploratório, observou-se incidência de outliers principalmente em `tcp.dstport` e `flow.duration`, além de menor impacto em outras métricas de fluxo. Esse diagnóstico orientou a manutenção de uma etapa de tratamento robusto no pré-processamento, preservando o comportamento operacional do tráfego IoT sem eliminar indevidamente padrões úteis para discriminação de ataques.

\subsection{Síntese do que foi feito}
- Seleção de 3 datasets IoT do universo TabularIoTAttack-2024.
- Consolidação e harmonização de rótulos em uma taxonomia final de segurança.
- Construção de artefato único multiclasse para treino/avaliação.
- Análise estatística (dimensões, tipos, classes, correlação, outliers, valores ausentes).
- Aplicação de PLS supervisionado (8 componentes) + transformação quantílica.
- Geração de base derivada pronta para pipeline QML/ML (`PC_1..PC_8` + labels).
