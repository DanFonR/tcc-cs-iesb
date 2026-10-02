OLS_VARIANTS :=

FLAGS_o0 := -O0
FLAGS_o2 := -O2
FLAGS_native := -O3 -march=native

all : main

define OLS_RULE
ols_$1_$2 :
	@ echo $$@
	@ $2 $$(FLAGS_$1) -Iinclude -Dols_linear_regression=$$@ \
	-c ./src/naive.c -o build/$$@.o \
	$(if $(filter clang,$2),-Rpass=loop-vectorize,-fopt-info-vec)
endef

define ADD_OLS
OLS_VARIANTS += ols_$1_$2
$$(eval $$(call OLS_RULE,$1,$2))
endef

$(foreach flag,o0 o2 native parallel,\
	$(foreach comp,clang gcc,\
		$(eval $(call ADD_OLS,$(flag),$(comp)))\
	)\
)

main : $(OLS_VARIANTS) $(HQR_VARIANTS)
	@ clang src/main.c build/* -Iinclude -o bin/main

hqr_base_gcc :
	@ echo $@
	@ gcc -O0 -Iinclude -Dlapack_linear_regression=$@ \
	-c ./src/householder.c -o ./build/$@.o

hqr_native_gcc :
	@ echo $@
	@ gcc $(FLAGS_native) -Iinclude -Dlapack_linear_regression=$@ \
	-c ./src/householder.c -o ./build/$@.o -fopt-info-vec

clean :
	@ rm -f build/*.o bin/main

