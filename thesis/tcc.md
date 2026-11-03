# 1. Introdução

<!-- Parágrafo de Contextualização -->
O desempenho computacional constitui um aspecto relevante na execução de algoritmos numéricos e científicos, especialmente em aplicações de aprendizado de máquina, cuja evolução tem sido acompanhada por uma crescente demanda por recursos computacionais ([Thompson et al., 2020](#tempo-aceitavel)). Nesse contexto, a regressão linear pelo método dos mínimos quadrados ordinários representa um problema adequado para investigar a relação entre a implementação de algoritmos, a eficiência de sua execução, e a precisão dos resultados, permitindo contrastar diferentes métodos numéricos e suas características. Além da escolha do método numérico, o desempenho de um programa compilado pode ser influenciado pelas otimizações realizadas pelo compilador. Os compiladores otimizadores, como o GCC e o Clang, conseguem analisar o código-fonte e aplicar transformações como *inlining* de funções e vetorização ou desenrolamento de laços para melhorar o tempo de execução do programa. Além disso, determinadas opções de compilação permitem sacrificar a conformidade estrita com o padrão IEEE 754 para aritmética de ponto flutuante ([IEEE, 2019](#i3e-floats)) em troca de ganhos de velocidade, o que pode causar perda de precisão dos resultados dos cálculos ([GNU Project, 2026](#otimizacoes-tempo)).

<!-- Parágrafo de Problema de Pesquisa -->
Logo, há a necessidade de avaliar o quanto, para uma dada arquitetura, compiladores modernos conseguem reduzir a diferença de desempenho entre uma implementação própria,  cujas otimizações dependem do compilador, e uma implementação de biblioteca especializada, que pode explorar de maneira mais eficiente os recursos do *hardware* disponível. Para esse estudo de caso, será avaliado o desempenho dos compiladores GCC e Clang, para linguagem C, na otimização da regressão linear pelo método das equações normais a diferentes níveis de otimização, tomando-se como base de comparação operações LAPACK da biblioteca OpenBLAS, como a decomposição QR de Householder, na plataforma AMD Zen 5, em ambiente Linux, conforme variam os tamanhos de entrada. Como se trata de uma avaliação de métodos numéricos, além do tempo de execução, para cada configuração de teste, tanto a precisão das previsões e dos coeficientes quanto a estabilidade numérica dos métodos empregados serão escrutinadas, para determinar uma possível associação entre a velocidade da implementação e a qualidade de seus resultados. A relevância deste estudo está na investigação dos limites das otimizações automáticas de compiladores diante de implementações numéricas especializadas.

<!-- Parágrafo de Objetivos e Estrutura -->
O objetivo deste trabalho é avaliar o desempenho de uma implementação de regressão linear por equações normais compilada com GCC e Clang em diferentes níveis de otimização, utilizando a decomposição QR de Householder da OpenBLAS como referência. Para isso, serão comparados os tempos de execução e a qualidade numérica das soluções para diferentes tamanhos de entrada, considerando também as otimizações aplicadas pelos compiladores e os recursos da arquitetura AMD Zen 5. O restante deste trabalho está organizado da seguinte forma: a Seção 2 apresenta o referencial teórico e estudos correlatos; a Seção 3 descreve a metodologia experimental; a Seção 4 apresenta e discute os resultados; e a Seção 5 reúne as conclusões do estudo.

# 2. Referencial Teórico
<!-- 1p minimos quadrados, 1p condicionamento, 1p para eq. normais, 1p para householder, 1p para am5, 1p para gcc/clang (mencionar avx-512), 1p para lapacke + openblas, 2-3p estado da arte, 1p sintese -->
## 2.1 Regressão Linear e seus Métodos

Os dois métodos a serem comparados têm como objetivo estimar uma série de coeficientes $\beta$ tal que, multiplicados para uma série de valores em $X$, aproxime o valor respectivo em $y$, tal que se minimize a soma dos quadrados da diferença entre o valor real e o valor previsto ([Chein, 2019](#definicao-mqo)). Para tanto, existem algumas maneiras de determinar os coeficientes, e o método mais ensinado, e um dos dois a ser comparado, é por equações normais, na qual determina-se o vetor de coeficientes pela resolução da equação $X^T X \beta = X^T y$. Porém, a formação de $X^TX$ eleva o número de condicionamento ao quadrado, isso é, a sua sensibilidade a erros ([Higham, 2020a](#condicionamento)), o que torna o método de equações normais menos numericamente estável, o que faz com que erros pequenos possam aumentar significativamente e distorcer o resultado final dos coeficientes ([Higham, 2020b](#estabilidade)).

Por isso, o método de decomposição QR por tranformações de Householder torna-se mais atraente, pois não há formação explícita de $X^TX$, e a matriz $X$ é decomposta em uma matriz ortonormal $Q$ e uma triangular superior $R$, tal que, ao se desenvolver as substituições, o sistema linear a ser resolvido torna-se $R\beta = Q^Ty$. Essa decomposição é usada na implementação de regressores lineares, por exemplo, na linguagem R, com a função lm ([R Core Team, s.d.](#r-lm)), ou em LAPACK com a família XGELS de funções, como DGELS ([LAPACK, s.d.](#lapack-dgels)). Apesar disso, esse método pode ser mais lento do que por equaçõs normais por exigir mais operações aritméticas em certas ocasiões ([Golub; Van Loan, 2013](#qr-velocidade)). Um dos objetivos do trabalho é justamente observar essa diferença de estabilidade e velocidade.

## 2.2 Compiladores, Níveis de Otimização, e Bibliotecas Especializadas

O GCC e o Clang estão entre os compiladores de linguagem C mais populares ([Jetbrains, 2025](#popularidade)), possuindo opções, ou *flags*, de otimização tanto para velocidade quanto para tamanho do executável. Algumas delas são agrupadas sob as opções -O, que variam de -O0, a padrão, que desativa as otimizações, até -O3, que habilita otimizações mais agressivas para tentar aumentar a velocidade do programa (a custo de velocidade de compilação), como a permuta de laços (*loop interchange*) ou a vetorização dos mesmos ([Free Software Foundation, [s. d.]](#gcc-optim); [LLVM Project, [s. d.]](#clang-optim)). Os recursos de arquitetura específicos da CPU alvo, como extensões AVX-512, podem ser usados, em conjunto com a opção -O3, por meio da opção -march=native, para permitir mais algumas otimizações, em troca de portabilidade.

Apesar disso, muito ainda depende da implementação algorítmica e, para esse fim, bibliotecas como a OpenBLAS, que implementa rotinas das especificações BLAS e LAPACK, fornecem funções altamente otimizadas para a realização de operações envolvendo álgebra linear. Diferentemente de uma implementação própria, essas bibliotecas podem explorar otimizações específicas para determinadas arquiteturas de processadores, incluindo o uso de instruções vetoriais e técnicas de paralelização, além de estratégias que reduzem o acesso à memória e melhoram o aproveitamento da hierarquia de cache. Dessa forma, podem alcançar níveis de desempenho difíceis de reproduzir apenas com as otimizações automáticas de um compilador. Neste trabalho, a OpenBLAS, será utilizada como referência de desempenho, por meio de rotinas LAPACK para a resolução de sistemas de mínimos quadrados pela decomposição QR, usando a interface C LAPACKE ([OPENBLAS, [s.d.]](#openblas)).

## 2.3 Arquitetura AMD Zen 5

Para a realização deste trabalho, é importante utilizar uma arquitetura moderna, com recursos relevantes para a execução de aplicações numéricas. Nesse contexto, a microarquitetura AMD Zen 5, lançada em 2024 para processadores de computadores pessoais e servidores, atende a esses requisitos. Entre suas características, destacam-se a capacidade de prever até dois desvios tomados por ciclo de clock, seis unidades lógicas aritméticas para operações inteiras e, por núcleo, 80 KB de cache L1, 48 KB dos quais para dados, e 1MB de cache L2, com cache L3 global mínimo de 32 MB. Além disso, a arquitetura possui suporte a AVX-512 com caminhos de dados de ponto flutuante de 512 bits, permitindo a execução de operações vetoriais com maior largura de dados ([AMD, 2024](#zen5)). Essa última característica é de bastante interesse para os experimentos de otimização, visto que ao menos parte dos laços poderão ser vetorizados, possivelmente com instruções dessa extensão. Além disso, a capacidade das memórias cache da arquitetura permite manter uma parcela significativa dos conjuntos de dados, especialmente os menores, próximos às unidades de processamento, reduzindo a necessidade de acessos à memória principal. Esse aspecto pode favorecer o desempenho das operações matriciais.

## 2.4 Trabalhos Correlatos

[Larabel (2026)](#benchmark) comparou o desempenho dos compiladores GCC 16 e LLVM/Clang 22 utilizando um processador AMD Ryzen Threadripper 9980X, baseado na arquitetura Zen 5, com as opções de otimização -O3 -march=native. Os testes abrangeram diferentes aplicações, como processamento de imagens, codificação de vídeo, rastreamento de raios e processamento digital de sinais. Os resultados mostraram diferenças de desempenho entre os compiladores, sem que um deles apresentasse vantagem em todas as aplicações avaliadas. Na biblioteca Liquid-DSP, por exemplo, o Clang obteve resultados superiores em diversas configurações. Essa variação mostra que a escolha do compilador pode influenciar o desempenho de acordo com as características da aplicação, algo possivelmente relevante para os experimentos com algoritmos numéricos deste trabalho. <!-- <!> -->

Relacionado ao desempenho de compiladores, [Feng, He e Tao (2021)](#eval-vetorizacao) investigaram, com o *benchmark* TSVC (*Test Suite for Vectorizing Compilers*), os limites dos compiladores ICC, GCC e Clang quanto à capacidade de vetorizar e a melhoria de desempenho após a vetorização. Descobriram que, para essa suíte, o GCC (versões 5.5.0 a 9.2.0) conseguiu vetorizar 43,7% dos testes, enquanto o Clang (versões 5.0.1 a 9.0.0) conseguiu 32,4%, e que a capacidade de vetorização depende, dentre outros fatores, da existência de desvios condicionais e de dependências entre *arrays* e entre iterações, e, por isso, operações de redução ou programas sem dependências impeditivas são mais propensos a serem vetorizados, e algumas transformações como a permuta de laços pode ajudar facilitar essa operação. Portanto, características dos laços e análise de dependência influenciam na capacidade de vetorização dos compiladores.

[Chang, Kang e Titley-Peloquin (2020)](#ne-qr) reafirmam [Baboulin et. al (2009)](#custo-qr-normal) ao demonstrarem que as equações normais podem exigir até aproximadamente metade das operações de ponto flutuante comparado à decomposição QR quando o número de observações é muito maior que o número de variáveis. Em contrapartida, elas possuem maior sensibilidaede a erros pois dependem do quadrado do nível de condicionamento, enquanto que o método por fatoração QR possui dependência linear. Logo, há necessidade de ponderar custo computacional e precisão da solução ao escolher a implementação da regressão linear.

## 2.5 Síntese do Referencial

Os estudos indicam que o desempenho da regressão linear depende do método numérico empregado e das possíveis otimizações pelo compilador, que pode ou não aproveitar recursos do *hardware*. As equações normais são menos custosas computacionalmente, porém são menos estáveis, enquanto que a decomposição QR possui características opostas. A escolha de compilador, entre GCC e Clang, pode influenciar na *performance* de uma mesma implementação devido à capacidade de otimizá-la, por reordenação e vetorização dos laços ou uma melhor análise de dependência, por exemplo. Como a OpenBLAS, uma biblioteca madura, oferece a decomposição QR, que pode ser usada como implementação de referência, e que recursos da arquitetura AMD Zen 5 permitem possibilidades adicionais de otimização pelos compiladores e que uma parcela considerável dos dados esteja em cache, há base teórica suficiente para a realização dos experimentos a seguir.

# 3. Metodologia

## 3.1 Ambiente Experimental

Os experimentos foram realizados em um computador com um AMD Ryzen 9 9950X3D, da arquitetura Zen 5, no qual foi feito um *undervolt* global de 20 mV, e 128 GB de RAM DDR5, com clock de 6400 MT/s e latência CL 42. Quanto ao ambiente de execução, usou-se o Ubuntu 26.04.1 LTS, com kernel 6.18.33.2-microsoft-standard-WSL2, no WSL 2, e plano de energia "Performance" nas configurações do Windows 11 25H2. Para criação dos experimentos, usou-se o Clang 21 1:21.1.8-6ubuntu1, o GCC 15 15.2.0-16ubuntu1, GNU Make 4.4.1-3, liblapacke 3.12.1-7ubuntu1, e libopenblas-dev 0.3.32+ds-5.

Para o experimento principal, usou-se *double* como tipo de dado para os valores das matrizes, dispostas em ordem de linha, com ponteiro *restrict*, pois pode ajudar o compilador a realizar algumas otimizações como reordenar operações ou permitir vetorização, enquanto o experimento de estabilidade usa *long double* para cálculo mais preciso de erros. Para as opções de compilação, habilitaram-se relatórios de vetorização e substituição do nome da função de regressão por equações normais para cada configuração, para que todas as versões fossem acessíveis ao mesmo tempo.

Os dados foram gerados de forma sintética e pseudoaleatória, de modo a modelar um problema de regressão linear múltipla $y = X\beta + \varepsilon$. Utilizou-se o gerador xoshiro256** ([Blackman; Vigna, 2021](#rand-xoshiro)) com semente fixa (42), inicializado por splitmix64 ([Steele Jr.; Lea; Flood, 2014](#rand-split)), o que torna os conjuntos reprodutíveis. Para cada número de variáveis independentes $p \in \{10, 20, …, 100\}$, acrescido do intercepto (coluna de uns), foi gerada uma matriz de projeto com $10^6$ amostras e um vetor de coeficientes $\beta$ com entradas uniformes em $[0, 1)$; a resposta foi calculada como $y = X\beta + \varepsilon$, com $\varepsilon$ uniforme em $[−10^{-3}, 10^{-3})$. Dois tipos de conjunto foram produzidos: bem condicionado, com colunas independentes e uniformes em $[0, 1)$ (número de condição $\kappa_2(X)$ entre 12 e 95), e mal condicionado, em que cada coluna de índice par é a coluna anterior acrescida de ruído uniforme em $[−10^{-3}, 10^{-3})$, resultando em colunas quase colineares ($\kappa_2(X)$ entre aproximadamente $4,8\cdot10^3$ e $1,6\cdot10^4$). Os subconjuntos de $10^3$, $10^4$, $10^5$ e $10^6$ amostras correspondem às primeiras linhas de cada arquivo.

Para o experimento de estabilidade numérica, foram gerados conjuntos adicionais com $10^4$ amostras e $p \in \{10, 50, 100\}$, variando a amplitude do ruído de colinearidade em $10^{-3}$, $10^{-5}$, $10^{-6}$ e $10^{-7}$, além do caso bem condicionado, o que cobre $\kappa_2(X)$ de cerca de $10$ a $10^8$. Os dados são gravados em arquivos binários (número de amostras, número de variáveis, $\beta$, y e X em ordem de linhas) antes das medições e carregados em memória antes de cada bloco de testes.

## 3.2 Implementações

As implementações usadas no experimento principal foram o método de equações normais, com formação explícita de $X^TX$ e $X^Ty$, e uma que utiliza funções LAPACK da biblioteca OpenBLAS, que serviu como base para comparações. O primeiro método usou decomposição de Cholesky para se criar uma matriz triangular inferior, e substituição progressiva e regressiva para determinar o vetor de coeficientes. Já pelo segundo método, fez-se a decomposição QR de Householder da matriz X, seguida de uma sobrescrita pela multiplicação da transposta da matriz ortogonal Q ($y \gets Q^Ty$) e, por fim, a resolução do sistema triangular resultante ($R\beta = Q^Ty$), que retorna o vetor de coeficientes. Em ambos os casos, faz-se a medição de tempo de cada etapa e determina-se os resíduos $y_{true} - y_{pred}$.

A implementação por equações normais foi compilada seis vezes conforme o produto $\{\text{Clang}, \text{GCC}\} \times \{\text{-O0}, \text{-O1}, \text{-O3 -march=native}\}$, enquanto a implementação em OpenBLAS foi compilada somente com GCC em nível O0. Para realizar comparações de forma justa, todas as configurações usam somente uma *thread*.

## 3.3 Métricas e procedimento de análise

Para cada configuração, fez-se 5 rodadas de aquecimento e 20 rodadas de medição, na qual foram medidos o número de condicionamento, tempo de formação das matrizes $X^TX$ e $X^Ty$ (no caso do método das equações normais), tempo de decomposição (por método de Cholesky ou por QR), tempo de solução após a decomposição, as normas dos erros dos coeficientes e dos resíduos, tanto absolutos quanto relativos aos valores verdadeiros, o erro absoluto médio e a raíz do erro quadrático médio deles, e o mínimo e máximos absolutos.

Com essas métricas, o *speedup* das 6 configurações relativo à implementação com OpenBLAS pôde ser determinado, e ele e as outras métricas puderam ser analisadas em Python com as bibliotecas Matplotlib e pandas. Como as compilações do método de equações normais possuíam relatórios de vetorização, eles também foram analisados.

# 4. Resultados e Discussão

## 4.1 Relatórios de Vetorização

<!--  -->

## 4.2 Resultados de Tempo

<!--  -->

## 4.3 Resultados Numéricos

<!--  -->

# 5. Conclusão

<!--  -->

# Referências

<p id="tempo-aceitavel">
	THOMPSON, Neil C.; GREENEWALD, Kristjan; LEE, Keeheon; MANSO, Gabriel F. The computational limits of deep learning. arXiv, 2020. Disponível em: https://arxiv.org/abs/2007.05558. Acesso em: 20 out. 2026.
</p>

<p id="i3e-floats">
	INSTITUTE OF ELECTRICAL AND ELECTRONICS ENGINEERS (IEEE). IEEE Standard for Floating-Point Arithmetic. New York: IEEE, 2019. IEEE Std 754-2019.
</p>

<p id="otimizacoes-tempo">
	GNU PROJECT. Optimize Options (Using the GNU Compiler Collection (GCC)). Disponível em: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html. Acesso em: 22 out. 2026.
</p>

<p id="rand-split">
	STEELE JR., Guy L.; LEA, Doug; FLOOD, Christine H. Fast splittable pseudorandom number generators. In: ACM SIGPLAN INTERNATIONAL CONFERENCE ON OBJECT-ORIENTED PROGRAMMING, SYSTEMS, LANGUAGES, AND APPLICATIONS, 2014, Portland. Proceedings [...]. New York: ACM, 2014. p. 453–472.
</p>

<p id="rand-xoshiro">
	BLACKMAN, David; VIGNA, Sebastiano. Scrambled linear pseudorandom number generators. ACM Transactions on Mathematical Software, v. 47, n. 4, art. 36, p. 1–32, 2021.
</p>

<p id="definicao-mqo">
    CHEIN, Flávia. Introdução aos modelos de regressão linear: um passo inicial para compreensão da econometria como uma ferramenta de avaliação de políticas públicas. Brasília, DF: Escola Nacional de Administração Pública (Enap), 2019. 76 p. ISBN 978-85-256-0115-5.
</p>

<p id="condicionamento">
    HIGHAM, Nick. What is a Condition Number? Nick Higham, 19 mar. 2020a. Disponível em: https://nhigham.com/2020/03/19/what-is-a-condition-number/. Acesso em: 29 out. 2026.
</p>

<p id="estabilidade">
    HIGHAM, Nick. What is Numerical Stability? Nick Higham, 4 ago. 2020b. Disponível em: https://nhigham.com/2020/08/04/what-is-numerical-stability/. Acesso em: 29 out. 2026.
</p>

<p id="r-lm">
    R CORE TEAM. R: A language and environment for statistical computing. R Documentation: Fitting Linear Models. Disponível em: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html . Acesso em: 31 out. 2026.
</p>

<p id="lapack-dgels">
    LAPACK. DGELS: solves overdetermined or underdetermined systems for GE matrices. Netlib. Disponível em: https://netlib.org/lapack/lapack_routine/dgels.f . Acesso em: 31 out. 2026.
</p>

<p id="qr-velocidade">
    GOLUB, Gene H.; VAN LOAN, Charles F. Matrix Computations. 4. ed. Johns Hopkins University Press, 2013.
</p>

<p id="popularidade">
    JETBRAINS. The State of C 2025. 2025. Disponível em: https://lp.jetbrains.com/the-state-of-c-2025/ . Acesso em: 31 out. 2026.
</p>

<p id="gcc-optim">
    FREE SOFTWARE FOUNDATION. Optimize options (Using the GNU Compiler Collection). Disponível em: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html . Acesso em: 1 nov. 2026.
</p>

<p id="clang-optim">
    LLVM PROJECT. Clang: the Clang C, C++, and Objective-C compiler. Disponível em: https://clang.llvm.org/docs/CommandGuide/clang.html . Acesso em: 1 nov. 2026.
</p>

<p id="openblas">
    OPENBLAS. Developer Manual. OpenMathLib. Disponível em: https://www.openmathlib.org/OpenBLAS/docs/developers/ . Acesso em: 1 nov. 2026.
</p>

<p id="zen5">
    AMD. Zen 5 Delivers a New Level of Balanced Throughput. In: HOT CHIPS 2024. 2024. Disponível em: https://hc2024.hotchips.org/assets/program/conference/day2/24_HC2024.AMD.Cohen.Subramony.final.pdf . Acesso em: 2 nov. 2026.
</p>

<p id="custo-qr-normal">
    BABOULIN, Marc; GIRAUD, Luc; GRATTON, Serge; LANGOU, Julien. Parallel tools for solving incremental dense least squares problems: application to space geodesy. Journal of Algorithms & Computational Technology, v. 3, n. 1, p. 117–140, 2009.
</p>

<p id="benchmark">
    LARABEL, Michael. GCC 16 produces faster binaries than GCC 15, competitive race with LLVM Clang 22. Phoronix, 13 maio 2026. Disponível em: https://www.phoronix.com/review/gcc-16-vs-clang-22 . Acesso em: 2 nov. 2026.
</p>

<p id="eval-vetorizacao">
    FENG, Jing Ge; HE, Ye Ping; TAO, Qiu Ming. Evaluation of Compilers’ Capability of Automatic Vectorization Based on Source Code Analysis. Scientific Programming, v. 2021, art. 3264624, 2021.
</p>

<p id="ne-qr">
    CHANG, Xiao-Wen; KANG, Peng; TITLEY-PELOQUIN, David. Error bounds for computed least squares estimators. Linear Algebra and its Applications, v. 586, p. 28–42, 2020.
</p>

<p id="placeholder">
    <!--  -->
</p>
