\section{Base de dados}
A base de dados utilizada nesse trabalho foi construída a partir de um recorte IoT do ecossistema CIC-BCCC-NRC TabularIoTAttack-2024, consolidando três coleções principais: `CIC-BCCC-NRC-Edge-IIoTSet-2022`, `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` e `CIC-BCCC-NRC-UQ-IOT-2022`. Entre essas, o Edge-IIoTset é uma das fontes centrais e descreve um cenário realista de cibersegurança em aplicações IoT/IIoT, concebido para avaliar sistemas de detecção de intrusões baseados em aprendizado de máquina. O testbed que o origina é estratificado em sete camadas (percepção IoT/IIoT, computação de borda, SDN, fog, blockchain, NFV e nuvem), integrando protocolos de referência (como thingsboard, mosquitto/MQTT, TCP/IP, ONOS, OPNFV, Hyperledger Sawtooth e gêmeos digitais) a fim de refletir requisitos operacionais de IoT/IIoT. Os dados foram gerados para além de 10 tipos de dispositivos (sensores de temperatura/umidade, ultrassônicos, nível d'água, pH, umidade do solo, frequência cardíaca, chama, entre outros) e anotados frente a múltiplos ataques.

Os datasets de origem são disponibilizados em formatos distintos, de pacote (.pcap) e tabular (.csv). Devido a isso, podemos realizar a integração de datasets de diferentes origens, dado sua comum origem no formato (.pcap). Assim, podemos utilizar a metologia \it{CICflowMeter} para transformá-los em formato tabular (.csv) de similar formato. Portanto, temos os datasets mergidos, resultando no espaço de classes de 22 distintos rótulos de ataque e um total de 14.860.312 linhas. Após a harmonização das classes, a base final passou a ser descrita por um conjunto padronizado de atributos.

Para fins de análise estatística exploratória, foi utilizado um recorte com 200.000 registros e 85 colunas, sendo 80 variáveis numéricas, 5 categóricas (`flow.id`, `ip.src_host`, `ip.dst_host`, `frame.time`, `Attack_label`) e 4 variáveis binárias, incluindo `Attack_type`. Esse recorte permitiu caracterizar distribuição de classes, estatísticas descritivas, correlação e incidência de outliers sobre o conjunto efetivamente utilizado nos experimentos.

Para melhor organização das nomenclaturas e explicação dos tipos de ataques, o Quadro~\ref{ref:quadroExplicacaoAtaque} foi elaborado. 

\begin{table}[H]
\centering
\caption{Referência dos ataques e suas respectivas descrições}
\label{ref:quadroExplicacaoAtaque}
\begin{tabular}{|c|c|p{8.5cm}|}
\hline
\textbf{Ref.} & \textbf{Ataque} & \textbf{Descrição} \\
\hline
\hline
A0  & Backdoor              & Acesso remoto não autorizado via porta ou código oculto. \\
\hline
A1  & DDoS\_HTTP            & Sobrecarga de servidor web com requisições HTTP para indisponibilizá‑lo. \\
\hline
A2  & DDoS\_ICMP            & Envio massivo de pacotes ICMP (ping) para saturar a rede do alvo. \\
\hline
A3  & DDoS\_TCP             & Consumo de recursos por flood de conexões/segmentos TCP. \\
\hline
A4  & DDoS\_UDP             & Flood de pacotes UDP destinado a esgotar largura de banda ou serviços. \\
\hline
A5  & Fingerprinting        & Coleta de informações sobre sistema/serviços para identificar vetores de ataque. \\
\hline
A6  & MITM                  & Interceptação (e possível alteração) de comunicações entre duas partes. \\
\hline
A7  & Password              & Tentativas de obtenção de credenciais (força bruta, dicionário, engenharia social). \\
\hline
A8  & Ransomware            & Malware que cifra dados e exige resgate para recuperação. \\
\hline
A9  & SQL\_injection        & Inserção de código SQL malicioso para manipular ou extrair dados do BD. \\
\hline
A10 & Uploading             & Envio de arquivos maliciosos para comprometer sistemas ou executar código. \\
\hline
A11 & Port\_Scanning        & Varredura de portas para identificar serviços expostos e possíveis vulnerabilidades. \\
\hline
A12 & Vulnerability\_scanner& Ferramenta que detecta falhas conhecidas em sistemas ou aplicações. \\
\hline
A13 & XSS                   & Injeção de script malicioso em aplicações web para afetar navegadores de usuários. \\
\hline
A14 & Outras                & Categoria agregada para ataques fora do conjunto especialista (taxonomia final). Inclui, de forma explícita, os rótulos originais `ACK Flood` e `SYN Flood` (proveniência: `CIC-BCCC-NRC-UQ-IOT-2022`), bem como `DDoS PSHACK Flood` e `DDoS RSTFIN Flood` (proveniência: `CIC-BCCC-NRC-IoT-2023-Original Training and Testing`). Além disso, quaisquer ataques não mapeados ou não mencionados diretamente na política final de remapeamento são direcionados para `Others/Outras` para evitar corrupção silenciosa de rótulos e manter uma taxonomia estável para treinamento. \\
\hline
\end{tabular}
\end{table}



\subsection{Pré-processamento da base de dados}

Em uma primeira etapa, a base de dados passou por uma análise estatística descritiva, com inspeção dos tipos das variáveis, distribuição das classes e verificação de consistência do schema entre arquivos de origem. A partir dessa inspeção, foi entendido a necessidade de aplicar técnicas de limpeza e padronização, pois parte das colunas do conjunto de dados se mostrou corrompida, não interpretável ou com baixa utilidade estatística para modelagem. Um exemplo objetivo observado foi a existência de colunas constantes (sem variância), como `icmp.unused`, `http.tls_port`, `dns.qry.type`, `dns.retransmit_request_in`, `mqtt.msg_decoded_as`, `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`, que foram removidas por não agregarem poder discriminativo (mesmo valor em todas as amostras).

Adicionalmente, nesse mesmo estudo, houve a necessidade de excluir um conjunto maior de atributos (cerca de 34 colunas) devido à alta esparsidade, pois se evidenciou que estas apresentavam pelo menos 50\% dos seus valores iguais a zero. Na prática, esse critério foi utilizado para reduzir ruído, evitar que a escala/normalização fosse dominada por zeros estruturais e melhorar a estabilidade dos modelos, sobretudo nas etapas supervisionadas subsequentes.

Após a limpeza, os dados foram harmonizados para uma convenção única de nomes de colunas (`lowercase.dot`), reduzindo divergências semânticas entre fontes (por exemplo, mapeamentos equivalentes ao schema de referência de tráfego/wireshark quando aplicável), e os rótulos foram remapeados para a taxonomia final descrita na Seção anterior. O artefato final consolidado mantém 85 colunas (incluindo `Attack_label` e `Attack_type`) e é o ponto de partida para o pipeline de aprendizado.

\subsubsection{Remapeamento de rótulos e a classe \textit{Outras}}
Como as três coleções de origem possuem diferentes espaços de rótulos e diferentes granularidades para ataques volumétricos, foi necessário construir uma política explícita de remapeamento para uma taxonomia final estável. O tráfego benigno (`Benign Traffic`) foi mapeado para `NORMAL`. Parte dos ataques foi mapeada diretamente para as classes especialistas (por exemplo, `DDoS HTTP Flood` \(\rightarrow\) `DDoS_HTTP`, `SQL Injection` \(\rightarrow\) `SQL_injection`, `Vulnerability Scanner` \(\rightarrow\) `Vulnerability_scanner`), e também foram aplicadas fusões de aliases para reduzir fragmentação (por exemplo, `DDoS ACK Fragmentation` \(\rightarrow\) `DDoS_TCP`, `Telnet Brute Force` \(\rightarrow\) `Password`, `Recon Port Scan` \(\rightarrow\) `Port_Scanning`).

A classe `Outras` (ou `Others`) foi utilizada como classe guarda-chuva para ataques que não pertencem ao conjunto especialista (ou que não possuem correspondência semântica direta confiável na taxonomia final), incluindo explicitamente `ACK Flood`, `SYN Flood`, `DDoS PSHACK Flood` e `DDoS RSTFIN Flood`. Essa decisão tem dois objetivos: (i) evitar atribuições incorretas por similaridade superficial entre nomes; e (ii) garantir que o pipeline não descarte amostras de ataque, mantendo-as como tráfego malicioso rotulado, porém agregadas.

\subsubsection{Detecção outliers}

A identificação de anomalias em informações operacionais do Edge-IIoTset é crucial para remover variações extremas legítimas que acontecem em contextos industriais, antes de implementar modelos para detectar ataques. Para essa finalidade, foram utilizados dois métodos que se complementam: o Z-score, que é um método estatístico univariado, e o algoritmo Isolation Forest, destinado à análise multivariada utilizando aprendizado de máquina não supervisionado.

O Z-score ou Pontuação Padrão, uma medida numérica usada em estatística, é utilizada para avaliar a distância, em desvios padrão, entre um valor medido \( x \) e a média \( \mu \) da sua variável ~\cite{9754093}. O Z-score é demonstrado na Equação \ref{eq:zscore}. Valores cujo escore absoluto \( |Z| \) excede um limite estipulado são vistos como outliers.

\begin{equation}
Z = \frac{x - \mu}{\sigma}
\label{eq:zscore}
\end{equation}

Onde, $x$ é o valor real de uma variável no conjunto de dados, $\mu$ é a média de todos os valores da variável analisada e o $\sigma$ mede a dispersão dos dados em torno da média.

De forma complementar, foi empregado o algoritmo Isolation Forest, que modela a dificuldade de isolar amostras em partições aleatórias do espaço de atributos e, assim, fornece um score de anomalia em análise multivariada. Essa abordagem é particularmente útil em dados de tráfego de rede, pois combinações de variáveis podem caracterizar comportamentos raros que não são evidentes em uma inspeção univariada.

\subsubsection{Redução supervisionada de dimensionalidade via PLS}
Como etapa posterior ao preparo/limpeza e visando uma representação mais compacta (especialmente útil para pipelines que exigem dimensionalidade controlada), foi utilizada a técnica Partial Least Squares (PLS) como redução supervisionada. O procedimento adotado foi: (i) dividir o conjunto em treino e teste de forma estratificada; (ii) ajustar `PLSRegression` com 8 componentes utilizando apenas o treino (evitando \textit{data leakage}); (iii) transformar treino e teste em componentes `PC_1..PC_8`; e (iv) aplicar um mapeamento quantílico (\textit{QuantileTransformer}) para projetar os componentes para uma distribuição uniforme em \([0,1]\), novamente ajustada apenas no treino e reaplicada no teste. O resultado é uma base derivada com componentes supervisionados, apropriada para etapas de modelagem subsequentes.