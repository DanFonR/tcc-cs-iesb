.DEFAULT_GOAL := all
HEADERS := $(wildcard include/*.h)
LIBS    := -lopenblas -lm

FLAGS_o0     := -O0
FLAGS_o1     := -O1
FLAGS_native := -O3 -march=native

NE_OBJS :=
HQR_OBJS := build/hqr_base_gcc.o

define NE_RULE
NE_OBJS += build/ne_$1_$2.o
build/ne_$1_$2.o : src/naive.c $(HEADERS) | build
	@ echo $$@
	@ $2 $$(FLAGS_$1) -Iinclude -Dne_linear_regression=ne_$1_$2 \
	  -c $$< -o $$@ \
	  $(if $(filter clang,$2),-Rpass=loop-vectorize,-fopt-info-vec)
endef

$(foreach flag,o0 o1 native,\
  $(foreach comp,clang gcc,\
    $(eval $(call NE_RULE,$(flag),$(comp)))))

all : bin/main

build/hqr_base_gcc.o : src/householder.c $(HEADERS) | build
	@ echo $@
	@ gcc -O0 -Iinclude -Dlapack_linear_regression=hqr_base_gcc -c $< -o $@

bin/main : src/main.c src/tests.c $(NE_OBJS) $(HQR_OBJS) $(HEADERS) | bin
	@ clang -O2 -Iinclude src/main.c src/tests.c $(NE_OBJS) $(HQR_OBJS) \
	  -o $@ $(LIBS)

bin/datagen : src/datagen/datawriter.c | bin
	@ gcc -O2 $< -o $@

datagen : bin/datagen
	@ mkdir -p data/wellcond data/illcond data/results
	@ ./bin/datagen

build bin :
	@ mkdir -p $@

clean :
	@ rm -rf build bin/main

env_check :
	@ gcc -march=native -Q --help=target | grep -E '^ *-march='
	@ clang -march=native -x c -E /dev/null -### 2>&1 \
		| grep -o 'target-cpu" "[^"]*"'

.PHONY : all datagen clean env_check
