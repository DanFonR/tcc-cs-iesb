# 1. Introdução

## ¶. Contextualização
O desempenho computacional constitui um aspecto relevante na execução de algoritmos numéricos e científicos, especialmente em aplicações de aprendizado de máquina, cuja evolução tem sido acompanhada por uma crescente demanda por recursos computacionais ([Thompson et al., 2020](./referencias.md#tempo-aceitavel)).
Nesse contexto, a regressão linear pelo método dos mínimos quadrados ordinários representa um problema adequado para investigar a relação entre a implementação de algoritmos, a eficiência de sua execução, e a precisão dos resultados, permitindo contrastar diferentes métodos numéricos e suas características. Além da escolha do método numérico, o desempenho de um programa compilado pode ser influenciado pelas otimizações realizadas pelo compilador.
Os compiladores otimizadores, como o GCC e o Clang, conseguem analisar o código-fonte e aplicar transformações como *inlining* de funções e vetorização ou desenrolamento de laços para melhorar o tempo de execução do programa. Além disso, determinadas opções de compilação permitem sacrificar a conformidade estrita com o padrão IEEE 754 para aritmética de ponto flutuante ([IEEE, 2019](./referencias.md#i3e-floats)) em troca de ganhos de velocidade, o que pode causar perda de precisão dos resultados dos cálculos ([GNU PROJECT, 2026](./referencias.md#otimizacoes-tempo)).

## ¶. Problema de pesquisa
Logo, há a necessidade de avaliar o quanto, para uma dada arquitetura, compiladores modernos conseguem reduzir a diferença de desempenho entre uma implementação própria,  cujas otimizações dependem do compilador, e uma implementação de biblioteca especializada, que pode explorar de maneira mais eficiente os recursos do *hardware* disponível. Para esse estudo de caso, será avaliado o desempenho dos compiladores GCC e Clang, para linguagem C, na otimização da regressão linear pelo método das equações normais a diferentes níveis de otimização, tomando-se como base de comparação operações LAPACK da biblioteca OpenBLAS, como a decomposição QR de Householder, na plataforma AMD Zen 5, em ambiente Linux, conforme variam os tamanhos de entrada. Como se trata de uma avaliação de métodos numéricos, além do tempo de execução, para cada configuração de teste, tanto a precisão das previsões e dos coeficientes quanto a estabilidade numérica dos métodos empregados serão escrutinadas, para determinar uma possível associação entre a velocidade da implementação e a qualidade de seus resultados. A relevância deste estudo está na investigação dos limites das otimizações automáticas de compiladores diante de implementações numéricas especializadas.

## ¶. Objetivos e Estrutura
O objetivo deste trabalho é avaliar o desempenho de uma implementação de regressão linear por equações normais compilada com GCC e Clang em diferentes níveis de otimização, utilizando a decomposição QR de Householder da OpenBLAS como referência. Para isso, serão comparados os tempos de execução e a qualidade numérica das soluções para diferentes tamanhos de entrada, considerando também as otimizações aplicadas pelos compiladores e os recursos da arquitetura AMD Zen 5. O restante deste trabalho está organizado da seguinte forma: a Seção 2 apresenta o referencial teórico e estudos correlatos; a Seção 3 descreve a metodologia experimental; a Seção 4 apresenta e discute os resultados; e a Seção 5 reúne as conclusões do estudo.

# 2. Referencial Teórico
<!-- 1p minimos quadrados, 1p condicionamento, 1p para eq. normais, 1p para householder, 1p para am5, 1p para gcc/clang (mencionar avx-512), 1p para lapacke + openblas, 2-3p estado da arte, 1p sintese -->


# 3. Metodologia

## 3.1 Ambiente Experimental

Os experimentos foram realizados em um computador com um AMD Ryzen 9 9950X3D, da arquitetura Zen 5, no qual foi feito um *undervolt* global de 20 mV, e 128 GB de RAM DDR5, com clock de 6400 MT/s e latência CL 42. Quanto ao ambiente de execução, usou-se o Ubuntu 26.04.1 LTS, com kernel 6.18.33.2-microsoft-standard-WSL2, no WSL 2, e plano de energia "Performance" nas configurações do Windows 11 25H2. Para criação dos experimentos, usou-se o Clang 21 1:21.1.8-6ubuntu1, o GCC 15 15.2.0-16ubuntu1, GNU Make 4.4.1-3, liblapacke 3.12.1-7ubuntu1, e libopenblas-dev 0.3.32+ds-5.

Para o experimento principal, usou-se *double* como tipo de dado para os valores das matrizes, dispostas em ordem de linha, com ponteiro *restrict*, pois pode ajudar o compilador a realizar algumas otimizações como reordenar operações ou permitir vetorização, enquanto o experimento de estabilidade usa *long double* para cálculo mais preciso de erros. Para as opções de compilação, habilitaram-se relatórios de vetorização e substituição do nome da função de regressão por equações normais para cada configuração, para que todas as versões fossem acessíveis ao mesmo tempo.

Os dados foram gerados de forma sintética e pseudoaleatória, de modo a modelar um problema de regressão linear múltipla $y = X\beta + \varepsilon$. Utilizou-se o gerador xoshiro256** ([Blackman; Vigna, 2021](./referencias.md#rand-xoshiro)) com semente fixa (42), inicializado por splitmix64 ([Steele Jr.; Lea; Flood, 2014](./referencias.md#rand-split)), o que torna os conjuntos reprodutíveis. Para cada número de variáveis independentes $p \in \{10, 20, …, 100\}$, acrescido do intercepto (coluna de uns), foi gerada uma matriz de projeto com $10^6$ amostras e um vetor de coeficientes $\beta$ com entradas uniformes em $[0, 1)$; a resposta foi calculada como $y = X\beta + \varepsilon$, com $\varepsilon$ uniforme em $[−10^{-3}, 10^{-3})$. Dois tipos de conjunto foram produzidos: bem condicionado, com colunas independentes e uniformes em $[0, 1)$ (número de condição $\kappa_2(X)$ entre 12 e 95), e mal condicionado, em que cada coluna de índice par é a coluna anterior acrescida de ruído uniforme em $[−10^{-3}, 10^{-3})$, resultando em colunas quase colineares ($\kappa_2(X)$ entre aproximadamente $4,8\cdot10^3$ e $1,6\cdot10^4$). Os subconjuntos de $10^3$, $10^4$, $10^5$ e $10^6$ amostras correspondem às primeiras linhas de cada arquivo.

Para o experimento de estabilidade numérica, foram gerados conjuntos adicionais com $10^4$ amostras e $p \in \{10, 50, 100\}$, variando a amplitude do ruído de colinearidade em $10^{-3}$, $10^{-5}$, $10^{-6}$ e $10^{-7}$, além do caso bem condicionado, o que cobre $\kappa_2(X)$ de cerca de $10$ a $10^8$. Os dados são gravados em arquivos binários (número de amostras, número de variáveis, $\beta$, y e X em ordem de linhas) antes das medições e carregados em memória antes de cada bloco de testes.

## 3.2 Implementações

As implementações usadas no experimento principal foram o método de equações normais, com formação explícita de $X^TX$ e $X^Ty$, e uma que utiliza funções LAPACK da biblioteca OpenBLAS, que serviu como base para comparações. O primeiro método usou decomposição de Cholesky para se criar uma matriz triangular inferior, e substituição progressiva e regressiva para determinar o vetor de coeficientes. Já pelo segundo método, fez-se a decomposição QR de Householder da matriz X, seguida de uma sobrescrita pela multiplicação da transposta da matriz ortogonal Q ($y \gets Q^Ty$) e, por fim, a resolução do sistema triangular resultante ($R\beta = Q^Ty$), que retorna o vetor de coeficientes. Em ambos os casos, faz-se a medição de tempo de cada etapa e determina-se os resíduos $y_{true} - y_{pred}$.

A implementação por equações normais foi compilada seis vezes conforme o produto $\{\text{Clang}, \text{GCC}\} \times \{\text{-O0}, \text{-O1}, \text{-O3 -march=native}\}$, enquanto a implementação em OpenBLAS foi compilada somente com GCC em nível O0. Para realizar comparações de forma justa, todas as configurações usam somente uma *thread*.

## 3.3 Métricas e procedimento de análise

Para cada configuração, fez-se 5 rodadas de aquecimento e 20 rodadas de medição, na qual foram medidos o número de condicionamento, tempo de formação das matrizes $X^TX$ e $X^Ty$ (no caso do método das equações normais), tempo de decomposição (por método de Cholesky ou por QR), tempo de solução após a decomposição, as normas dos erros dos coeficientes e dos resíduos, tanto absolutos quanto relativos aos valores verdadeiros, o erro absoluto médio e a raíz do erro quadrático médio deles, e o mínimo e máximos absolutos.

Com essas métricas, o *speedup* das 6 configurações relativo à implementação com OpenBLAS pôde ser determinado, e ele e as outras métricas puderam ser analisadas em Python com as bibliotecas Matplotlib e pandas. Como as compilações do método de equações normais possuíam relatórios de vetorização, eles também foram analisados.

# 4. Resultados e Discussão
<!--  -->


# 5. Conclusão
<!--  -->


# Referências
<!--  -->
