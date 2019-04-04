/*
 *  PFSP.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#ifndef _PFSP_H__
#define _PFSP_H__

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "PBP.h"
using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::string;

class PFSP : public PBP
{
	
public:
	
	/*
	 * The number of jobs of the problem.
	 */
	int m_jobs;
	
	/*
	 * The number of machines of the problem.
	 */
	int m_machines;
	
	/*
	 * The processing times matrix.
	 */
	int **m_processingtimes;
    
    /*
     * The time table for the processing times.
     */
    int * m_timeTable;


	// The constructor. It initializes a flowshop scheduling problem from a file.
	PFSP();
	
    // The destructor.
    virtual ~PFSP();
	
	/*
	 * Reads and PFSP instance with the given filename.
	 */
	int Read(string filename);
	
	/*
	 * Evaluates the given solution with the total flow time criterion.
	 */
	double Evaluate(int * genes);

    /*
	 * Evaluates inverted solution of the given solution with the total flow time criterion.
	 */
    double EvaluateInv(int * genes);

    /*
     * Returns the size of the problem.
     */
    int GetProblemSize();

private: 
	
	
	
};
#endif





