g++ -c -fPIC bb_cpp.cc -o bb_cpp.o
g++ -shared -Wl,-soname,lib_run.so -o lib_run.so bb_cpp.o