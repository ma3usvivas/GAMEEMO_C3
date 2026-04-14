# README
## Desafio de ML Engineer para utilizar o dataset GAMEEMO e criar um modelo preditivo para analisar os sinais EEG relacionados a dados neurais.
### Nesse desafio, foi criado um modelo com os sinais de EEG para analisar e prever emoções. Para isso, foram criados 4 programas e uma interface StreamLit com FastAPI para prever os modelos.

## Dependências Necessárias (Acrescentar Versão):

sklearn, pandas, numpy, pickle, pytorch,

uvicorn, fastapi, streamlit

## Funcionamento:
### 1. Rodar o pre processamento

Primeiro, é necessário rodar o pré processamento dos dados para corrigir e criar novas features sobre os dados de EEG.

Abra PreProcessing.py e rode python PreProcessing.py no terminal

### 2. Rodar o LOSOCV

Em seguida, selecione o melhor modelo na busca de grade pelo melhor modelo

Abra LOSOCV.py e rode python LOSOCV.py no terminal

### 3. Seleção do Melhor Modelo

A partir daí selecione o modelo que achar que possuir melhor perfomance para a tarefa que preferir

Abra BestModel.py e rode python BestModel.py no terminal

Será gerado um txt mostrando as métricas dos modelos

### 4. Experimento de Ablação

Foi feito então o experimento de ablação para o modelo de XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.1)

Abra AblationStudy.py e rode python AblationStudy.py no terminal

### 5. Criação do modelo novo e uso da interface

Por fim, o modelo novo é criado com 

Assim, utilizando a pasta Final para carregar os modelos criados, faça no terminal:

Em InteracePlatform.py, rodar uvicorn API.InferencePlatform:app --reload

Por fim, em platformstreamlit.py, rodar streamlit run platformstreamlit.py

Para testar, envie a pasta compactada (zipada) para o modelo
