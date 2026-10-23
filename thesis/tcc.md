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
<!-- 1p minimos quadrados, 1p condicionamento, 1p para normais, 1p para householder, 1p para am5, 1p para gcc/clang (mencionar avx-512), 1p para lapacke + openblas, 2-3p estado da arte, 1p sintese -->


# 3. Metodologia
<!--  -->


# 4. Resultados e Discussão
<!--  -->


# 5. Conclusão
<!--  -->
