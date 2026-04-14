# Lições Aprendidas sobre o projeto

## 1. Não é viável fazer busca em grade com CNN

CNN não é um algoritmo feito para ter busca em grade, além do mais, existem otimizadores como ADAM que possuem learning_rate variável com o modelo, assim, é possível reduizir uma das dimensões da busca em grade.

## 2. A estrutura de um modelo de EEG é muito complexa e dependente de cada conjunto (conjunto subject-game por exemplo)

Como cada conjunto de variáveis, no exemplo do GAMEEMO, subject-game, possui informação relacionado a pouco tempo de modelo 

Assim, é preciso saber dividir quantas epochs de um conjunto sujeito-game, por exemplo deverão ser treinadas para não aumentar consideravelmente o tempo de treino ou não faltar informações e resultar em underfitting. 

## 3. Validação Cruzada LOSO é praticamente obrigatória para validar um modelo

Para que não haja vazamento nos dados entre cada sujeito, é necessário dividir o conjunto em treino e teste por sujeito. A partir daí gera um problema, visto que como a quantidade de sujeitos é pequena, mudar o split de treino e teste pode gerar mudanças bruscas. 

Logo, não é possível dividir em conjunto de treino/testes de maneira padrão para treinar um modelo. O modelo geraria uma variância significativa no resultado dependendo do split feito.

## 4. É possível reduzir o ablation por cada tipo de variável ao reduzir o número de sujeitos para LOSO

Nesse projeto, o ablation para um único modelo durou 4 horas. O ablation é feito para que as métricas relacionadas ao modelo sejam avaliadas, e verifiquem se estão ou não colaborando com o modelo. 

Dessa forma, mesmo se a métrica baseline, abaixar um pouco, o importante é medir a diferença entre a métrica baseline e a sem o atributo. 

## 5. A importância de um pipeline de EEG

Fica nítido a diferença na velocidade ao realizar um projeto ao utilizar um pipeline adaptável para testar EEG, criar um pipeline do zero, mesmo com ajuda de IA, é muito trabalhoso. Fazendo com que seja necessário escrita legível e escalável para realizar esse projeto. 

## 6. É necessário utilizar diferentes datasets para calcular as emoções

Após realizar o ablation, fica claro como cada emoção varia de acordo com as features do dataset. Comprovado cientificamente pelo fato de que cada parte do cérebro possui uma função diferente para cada emoção.
