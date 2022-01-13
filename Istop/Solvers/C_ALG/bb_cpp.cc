#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

int value = 0;

using namespace std;

 void test(bool** comp_matrix, short size){
        std::cout<<"got it "<<size<<std::endl;
        for (int i= 0; i< size; i++){
            std::cout<<comp_matrix[0][i]<< std::endl;
            }
        }



class Run{


    public:

        void step(bool* solution, bool* offers, double reduction,double*  reductions, bool** comp_matrix, short size){


            std::cout<<"here"<<std::endl;
        }

        void check(){
            std::cout<<"here"<<std::endl;
        }

        void test(bool** comp_matrix, short size){

        for (int i= 0; i< size; i++){
            std::cout<<comp_matrix[0][i]<< std::endl;
            }
        }



        ~Run() {}
};



extern "C" {
    Run* Run_()
    {return new Run(); }


    void check_(Run* run){ run -> check(); }
    void test_(bool** comp_matrix, short size) {test(comp_matrix, size);}
    //void step(bool* solution, bool* offers, double reduction,double*  reductions, double** comp_matrix, short size){ off -> print_triples(); }

}
