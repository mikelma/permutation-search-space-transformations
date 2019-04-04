/*
 *  PFSP.cpp
 *  CompetitionWCCI2014
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#include "PFSP.h"
#include "Tools.h"
#include "Variables.h"

/*
 *Class constructor.
 */
PFSP::PFSP()
{
	
}

/*
 * Class destructor.
 */
PFSP::~PFSP()
{
	for	(int i=0;i<m_machines;i++)
		delete [] m_processingtimes[i];
	delete [] m_processingtimes;
    delete [] m_timeTable;
    delete [] m_aux;
}

/*
 * Evaluates inverted solution of the given solution with the total flow time criterion.
 */
double PFSP::EvaluateInv(int * genes)
{
    Invert(genes,m_jobs,m_aux);
    
    for (int i=0;i<m_machines;i++) m_timeTable[i]=0;
	int j,z, job;
	int machine;
    int prev_machine=0;

	int first_gene=m_aux[0];
    m_timeTable[0]=m_processingtimes[0][first_gene];
	for (j=1;j<m_machines;j++)
	{
		m_timeTable[j]=m_timeTable[j-1]+m_processingtimes[j][first_gene];
	}
	
	double fitness=m_timeTable[m_machines-1];
	for (z=1;z<m_jobs;z++)
	{
		job=m_aux[z];
		
		//machine 0 is always incremental, so:
		m_timeTable[0]+=m_processingtimes[0][job];
		prev_machine=m_timeTable[0];
		for (machine=1;machine<m_machines;machine++)
		{
			m_timeTable[machine]= MAX(prev_machine,m_timeTable[machine])+ m_processingtimes[machine][job];
			prev_machine=m_timeTable[machine];
		}
		
		fitness+=m_timeTable[m_machines-1];
	}
	
	return -fitness;
}

/*
 * Evaluates the given solution with the total flow time criterion.
 */
double PFSP::Evaluate(int * genes)
{
    for (int i=0;i<m_machines;i++) m_timeTable[i]=0;
	int j,z, job;
	int machine;
    int prev_machine=0;
    
	int first_gene=genes[0];
    m_timeTable[0]=m_processingtimes[0][first_gene];
	for (j=1;j<m_machines;j++)
	{
		m_timeTable[j]=m_timeTable[j-1]+m_processingtimes[j][first_gene];
	}   
	
	double fitness=m_timeTable[m_machines-1];
	for (z=1;z<m_jobs;z++)
	{
		job=genes[z];
		
		//machine 0 is always incremental, so:
		m_timeTable[0]+=m_processingtimes[0][job];
		prev_machine=m_timeTable[0];
		for (machine=1;machine<m_machines;machine++)
		{
			m_timeTable[machine]= MAX(prev_machine,m_timeTable[machine])+ m_processingtimes[machine][job];
			prev_machine=m_timeTable[machine];
		}
		
		fitness+=m_timeTable[m_machines-1];
	}

	return -fitness;
}

int PFSP::Read(string filename)
{
    cout<<"reading..."<<endl;
	bool readMatrix=false;
	bool readDimension=false;

	char line[2048]; // variable for input value
	string data="";
	ifstream indata;
	indata.open(filename.c_str(),ios::in);
	while (!indata.eof())
	{
		//process lines
		indata.getline(line, 2048);
		stringstream ss;
		string sline;
		ss << line;
		ss >> sline;
        //cout<<"sline: "<<sline<<endl;
		if (strContains(line,"number of jobs")==true && readMatrix==true)
		{
			break;
		}
		else if (strContains(line,"number of jobs")==true)
		{
			readDimension=true;
		}
		else if (readDimension==true)
		{
			m_jobs = atoi(strtok (line," "));
			//cout<<"JOB NUM: "<<m_jobs<<endl;
			m_machines = atoi(strtok (NULL, " "));
			//cout<<"MACHINE NUM: "<<m_machines<<endl;
			readDimension=false;
			
		}
		else if (readMatrix)
		{;
			if (data=="")
				data = line;
			else
				data = data+' '+line;
		}
		else if (strContains(line,"processing times :"))
		{
			readMatrix=true;
		}
	}
	indata.close();
	
	//BUILD JOB PROCESSING MATRIX
	//cout << "--> BUILDING JOB PROCESSING MATRIX" << endl;
	m_processingtimes = new int*[m_machines];
	for (int i=0;i<m_machines;i++)
	{
		m_processingtimes[i]= new int[m_jobs];
	}
	m_timeTable= new int[m_machines];
	m_aux= new int[m_jobs];
	//FILL JOB PROCESSING MATRIX
	//cout << "--> FILLING JOB PROCESSING MATRIX: "<<data << endl;
	istringstream iss(data);
	int i=0;
	int j=0;
	do
	{
		string sub;
	    iss >> sub;
	    if (sub!="")
	    {
			//save distance in distances matrix. Save negative distance in order to minimize fitness instead of
			//maximize.
	    	m_processingtimes[i][j]= atoi(sub.c_str());
	    	if (j==m_jobs-1)
	    	{
	    		i++;
	    		j=0;
	    	}
	    	else
	    	{
	    		j++;
	    	}
	    }
	    else
	    {
	    	break;
	    }
	} while (iss);
	
	return (m_jobs);
}


/*
 * Returns the size of the problem.
 */
int PFSP::GetProblemSize()
{
    return m_jobs;
}

