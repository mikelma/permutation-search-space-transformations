/*
 *  QAP.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#ifndef _QAP_H__
#define _QAP_H__

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>

//Max operation.
#define MAX(A,B) ( (A > B) ? A : B)

//Min operation.
#define MIN(A,B) ( (A < B) ? A : B)

//Equal operation.
#define EQUAL(A,B) ( (A == B) ? 1 : 0)

//Not equal operation.
#define NON_EQUAL(A,B) ( (A != B) ? 1 : 0)

using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::stringstream;
using std::string;

class QAP
{
	
public:
	
    /*
     * The matrix of distances between the cities.
     */
	int ** m_distance_matrix;
	
    /*
     * The flow matrix.
     */
	int ** m_flow_matrix;
	
	/*
	 * The number of jobs of the problem.
	 */
	int m_size;
	
    /*
     * Average components of the elementary landscape decomposition.
     */
    float fc1_avg,fc2_avg,fc3_avg;
    
    long int m_totalpq;
	/*
     * The constructor. It initializes a QAP from a file.
     */
	QAP();
	
    /*
     * The destructor.
     */
    virtual ~QAP();
	
	/*
	 * Read QAP file.
	 */
	int Read(string filename);

	/*
	 * This function evaluates the individuals for the QAP problem.
	 */
	double Evaluate(int * genes);
        
    /*
     * Returns the size of the problem.
     */
    int GetProblemSize();
    
    /*
     * Calculates the fitness value corresponding to the first component of the elementary landscape decomposition.
     */
    double fc1(int * genes);
    
    /*
     * Calculates the fitness value corresponding to the second component of the elementary landscape decomposition.
     */
    double fc2(int * genes);
    double fc2_optimized(int * x);
    
    /*
     * Calculates the fitness value corresponding to the second component of the elementary landscape decomposition. This method is optimized for a more efficient computation.
     */
    double fc2_optimizedV2(int * x);
    
    /*
     * Calculates the fitness value corresponding to the third component of the elementary landscape decomposition.
     */
    double fc3(int * genes);
    
    /*
     * Calculates the fitness value decomposed in the three components of the elementary landscape decomposition.
     */
    void f_components(int * genes, double * components);
    
    /*
     * Calculates the average fitness components of the elementary landscape decomposition needed to calculate the average fitness of a neighborhood in close form.
     */
    void CalculateAverageComponents_ELD();
    

    
private:
	
};
#endif
