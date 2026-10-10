.DEFAULT_GOAL := all

# --- OpenBLAS (Debian/Ubuntu: pacote libopenblas-dev + liblapacke-dev) ---
ifeq ($(shell pkg-config --exists openblas && echo yes),)
$(error pkg-config nao encontrou o openblas; instale libopenblas-dev)
endif

BLAS_CFLAGS := $(shell pkg-config --cflags openblas)
# --no-as-needed: impede que o linker descarte o libopenblas e deixe o loader
# cair no BLAS/LAPACK de referencia. -lopenblas deve vir antes de -llapacke.
LIBS := -Wl,--no-as-needed $(shell pkg-config --libs openblas) -llapacke -lm

HEADERS := $(wildcard include/*.h)
WARN    := -Wall -Wextra

FLAGS_o0     := -O0
FLAGS_o1     := -O1
FLAGS_native := -O3 -march=native

NE_OBJS  :=
HQR_OBJS := build/hqr_base_gcc.o

# $1 = nivel de otimizacao (o0|o1|native), $2 = compilador (clang|gcc)
define NE_RULE
NE_OBJS += build/ne_$1_$2.o
build/ne_$1_$2.o : src/naive.c $(HEADERS) Makefile | build
	@ echo $$@
	@ $2 $(WARN) $$(FLAGS_$1) -Iinclude -Dne_linear_regression=ne_$1_$2 \
	  -c $$< -o $$@ \
	  $(if $(filter clang,$2),-Rpass=loop-vectorize -Rpass-missed=loop-vectorize,-fopt-info-vec -fopt-info-vec-missed)
endef

$(foreach flag,o0 o1 native,\
  $(foreach comp,clang gcc,\
    $(eval $(call NE_RULE,$(flag),$(comp)))))

all : bin/main bin/stability

build/hqr_base_gcc.o : src/householder.c $(HEADERS) Makefile | build
	@ echo $@
	@ gcc $(WARN) -O0 $(BLAS_CFLAGS) -Iinclude \
	  -Dlapack_linear_regression=hqr_base_gcc -c $< -o $@

bin/main : src/main.c src/tests.c $(NE_OBJS) $(HQR_OBJS) $(HEADERS) Makefile | bin
	@ clang $(WARN) -O2 $(BLAS_CFLAGS) -Iinclude src/main.c src/tests.c \
	  $(NE_OBJS) $(HQR_OBJS) -o $@ $(LIBS)

bin/stability : src/stability.c src/refsolve.c src/tests.c $(NE_OBJS) $(HQR_OBJS) $(HEADERS) Makefile | bin
	@ clang $(WARN) -O2 $(BLAS_CFLAGS) -Iinclude src/stability.c src/refsolve.c \
	  src/tests.c $(NE_OBJS) $(HQR_OBJS) -o $@ $(LIBS)

bin/datagen : src/datagen/datawriter.c Makefile | bin
	@ gcc $(WARN) -O2 $< -o $@

datagen : bin/datagen
	@ mkdir -p data/wellcond data/illcond data/stability data/results
	@ ./bin/datagen

# Roda o experimento de estabilidade numerica (gera data/results/stability.csv)
stability : bin/stability
	@ ./bin/stability

build bin :
	@ mkdir -p $@

clean :
	@ rm -rf build bin/main bin/stability

# Versao dos compiladores e CPU alvo reconhecida pelo -march=native
env_check :
	@ gcc --version | head -n 1
	@ clang --version | head -n 1
	@ pkg-config --modversion openblas
	@ gcc -march=native -Q --help=target | grep -E '^ *-march='
	@ clang -march=native -x c -E /dev/null
