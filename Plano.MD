#Sobre o projeto

- O objetivo do projeto é realizar a análise de sentimento de indivíduos enquanto eles assistiam vídeos de uma pessoa jogando (gameplay)

- O conjunto de dados mostra dados ao longo do tempo do EEG (eletroencefalograma) de 28 indivíduos (subjects), medindo o sinal elétrico da posição de eletrodos em pacientes. As regiões são marcadas como:
    F - Frontal 
    AF - Anterior Frontal
    FC - Fronto Central
    C - Central
    T - Temporal
    P - Pariental
    O - Occipital

E a numeração relacionada a posição do eletrodo na cabeça
Fontes: https://www.acns.org/UserFiles/file/Guideline2-GuidelinesforStandardElectrodePositionNomenclature_v1.pdf


- Além disso, os pacientes preencheram um formulário contendo o gênero (M ou F); a idade; perguntas como por exemplo se a pessoa possui problemas neurológicos e se ela já tinha jogado ou visto o jogo; o sentimento relacionado a cada um dos vídeos que foi assistido. Foi também feita uma avaliação sobre o quão feliz e o quão agitada a pessoa ficou vendo o vídeo.

- O sinal foi medido nos indivíduos enquanto assistiam vídeos de uma pessoa jogando (gameplay) 4 jogos distintos. Cada jogo, enumerado de 1 a 4, foi descrito pelo próprio conjunto de dados como: chato, calmo, horripilante e engraçado; respectivamente.

- Esses sinais e o formulário estão divididos em pastas com cada indivíduo tendo uma numeração XX (SXX), o qual dentro de cada pasta de individuo contem 3 pastas: dados pré-processados, dados crus (raw data) e as respostas dos formulários. Cada pasta de dados contém os dados do EEG dos indivíduos para cada gameplay assistida. Na pasta de formulários, estão os formulários descritos de cada jogo.

- Com essas informações, foi pedido para criar um modelo para classificar os sentimentos dos indivíduos de acordo com cada vídeo assistido.

#Plano de Ação
Plano de Ação para o projeto com o Database GAMEEMO

- Serão definidas tarefas e prazos que serão realizadas para definir o projeto.

##Limpeza dos Dados 04/04

- Primeiramente será feita a limpeza dos dados com a extração de ruído, remoção de outliers e normalização. Os dados "limpos" serão salvos em outra tabela com a finalidade de distinguir os dados e avaliar a diferença do resultado da métrica no modelo no experimento de ablação.

##Analise Exploratória de Dados - 05/04

- Em seguida será realizado o processo de análise exploratória de dados com o objetivo de extrair features principais relacionadas ao modelo.

##Criação de Modelo Inicial - 05/04

- Será criado um modelo inicial sem uma grande modificação dos recursos (features) com a finalidade de criar uma métrica como parâmetro. Nessa etapa, será feita uma das partes do expererimento de ablação 

## Criação de Modelo Final 09/04

- Nessa etapa será realizada a criação do modelo final, o qual os recursos (features) serão extraídos, a arquitetura final do modelo será definida e comparada com a arquitetura do modelo inicial, mais simples.

##Experimento de Ablação 10/04

- Nessa etapa, o modelo final será comparado com o modelo inicial, e serão avaliados quais são as características mais importantes do modelo.

##Criação da Plataforma de Inferência 12/04

- Será criado uma platafoma de inferência, o qual ela receberá os dados de um indivíduo e tentará prever o seu sentimento em relação ao jogo. É importante que nessa etapa, o modelo consiga receber dados de um indivíduo e retorná-los sem atraso e de forma escalável.

##Avaliação LOO K-Fold 12/04

- Nessa etapa, será feita a avaliação de LOO K-Fold para obter uma melhor estimativa do resultado da métrica. LOO K-Fold reduz a variância da avaliação e dá uma estimativa mais estável da performance real do modelo.

##Apresentação do Projeto 14/04

- Por fim, será realizada a apresentação do projeto apresentando o funcionamento do projeto e justificando todas as decisões tomadas para chegar no resultado final do projeto. 
