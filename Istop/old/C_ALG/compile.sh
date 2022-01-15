src=/home/andrea/Scrivania/Hotspot_/Istop/Solvers/C_ALG/
g++ -c -fPIC ${src}bb_cpp.cc -o ${src}bb_cpp.o
g++ -shared -Wl,-soname,${src}lib_run.so -o ${src}lib_run.so ${src}bb_cpp.o