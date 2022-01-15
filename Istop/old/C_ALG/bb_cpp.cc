#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <cstring>

int value = 0;

using namespace std;



class Run{

    bool* comp_vect;
    bool** comp_matrix;
    double* reductions;
    size_t size;

    bool initialSolution;
    double bestReduction;
    bool* sol;

    int nodes = 0;

    public:
    Run(bool* c_m, double* red, size_t s): comp_matrix{new bool*[s]}, reductions{new double[s]}, size{s}, comp_vect{new bool[s*s]}

    {
         memcpy(comp_vect, c_m, s*s*sizeof(bool));
         for (int i = 0 ; i< size; i++) {
                comp_matrix[i]= &comp_vect[i*size];
        }
        memcpy(reductions, red, s*sizeof(double));

        std::cout<<std::endl;
        initialSolution = false;
        bestReduction = 0;
        sol = new bool[size];
        for (int i= 0; i< size; i++){sol[i] = false;}

//        run();
    }

    void step(bool* solution, bool* offers, double reduction,double*  reds, bool** c_m){

        // leaf condition
        nodes += 1;


        int sum = 0;
        for (int i= 0; i< size; i++){
            sum += offers[i];
        }

        if (sum == 0){
            initialSolution = true;
            return;
        }

        // l_offers

        size_t idx = 0;
        while (offers[idx] == false) {idx++;}

        double l_reduction = reduction + reductions[idx];
        bool* l_solution = new bool[size];
        for (int i= 0; i< size; i++) {l_solution[i] = solution[i];}
        l_solution[idx] = true;


        // new optimal
        if (l_reduction > bestReduction){
            for (int i= 0; i< size; i++) {sol[i] = l_solution[i];};
            bestReduction = l_reduction;
        }


        bool* l_offers = new bool[size];
        for (int i= 0; i< size; i++) {l_offers[i] = comp_matrix[idx][i] * offers[i];}

        if (initialSolution) {
            double l_offers_reduction = 0;
            for (int i= 0; i< size; i++) {
                if (l_offers[i] == true) {l_offers_reduction += reductions[i];}
                }
            double bound = l_reduction + l_offers_reduction;

            if (bound > bestReduction){
                step(l_solution, l_offers, l_reduction, reductions, comp_matrix);
            }

        }
        else {step(l_solution, l_offers, l_reduction, reductions, comp_matrix);}


        // r_offers

        bool* r_offers = new bool[size];
        for (int i= 0; i< size; i++) {r_offers[i] = offers[i];}
        r_offers[idx] = false;

        double r_offers_reduction = 0;
        for (int i= 0; i< size; i++) {
            if (r_offers[i] == true) {r_offers_reduction += reductions[i];}
            }
        double bound = l_reduction + r_offers_reduction;
        if (bound > bestReduction){
            step(solution, r_offers, reduction, reductions, comp_matrix);
        }

        delete l_solution;
        delete l_offers;
        delete r_offers;


    }

    void run(){
        bool* solution = new bool[size];
        for (int i= 0; i< size; i++){solution[i] = false;}


        bool* offers = new bool[size];
        for (int i= 0; i< size; i++){offers[i] = true;}

        step(solution, offers, 0, reductions, comp_matrix);
        std::cout<<"reduction c "<<bestReduction<<size<<std::endl;


        delete solution;
        delete offers;


    }

     void print(){
        std::cout<<"Comp mat "<<size<<std::endl;
        for (int i= 0; i< size; i++){
            for (int j= 0; j< size; j++){
                std::cout<<comp_matrix[i][j]<<" ";
                }
                std::cout << std::endl;
            }


        std::cout<<std::endl<<"Reductions "<<size<<std::endl;
        for (int i= 0; i< size; i++){
             std::cout<<reductions[i]<<" ";
        }
        std::cout<<std::endl;
        }

     bool* get_solution() {return sol;}
     double get_reduction() {return bestReduction;}



    ~Run() {
        delete comp_matrix;
        reductions = nullptr;
        sol = nullptr;
        }
};



extern "C" {
    Run* Run_(bool* comp_matrix, double* reductions, size_t size)
    {return new Run(comp_matrix, reductions, size); }
    void run_(Run* run) {run -> run();}

    bool* get_solution_(Run* run) {return run -> get_solution();}
    double get_reduction_(Run* run) {return run -> get_reduction();}
    void print_(Run* run) {run -> print();}
    //void step(bool* solution, bool* offers, double reduction,double*  reductions, double** comp_matrix, size_t size){ off -> print_triples(); }

}
