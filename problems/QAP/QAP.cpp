/*
 *  QAP.cpp
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#include "QAP.h"


/*
 *Class constructor.
 */
QAP::QAP()
{

}

/*
 * Class destructor.
 */
QAP::~QAP()
{
	for (int i=0;i<m_size;i++)
	{
		delete [] m_distance_matrix[i];
		delete [] m_flow_matrix[i];
	}
	delete [] m_flow_matrix;
	delete [] m_distance_matrix;

}



int QAP::Read(string filename)
{
	char line[5096]; // variable for input value
	ifstream indata;
	indata.open(filename.c_str(),ios::in);
	int num=0;
	while (!indata.eof())
	{
		//LEER LA LINEA DEL FICHERO
		indata.getline(line, 5096);
		stringstream ss;
		string sline;
		ss << line;
		ss >> sline;
        //cout<<"line: "<<line<<endl;
		if (num==0)
		{
			//OBTENER EL TAMAÑO DEL PROBLEMA
			m_size = atoi(sline.c_str());
			m_distance_matrix = new int*[m_size];
			m_flow_matrix = new int*[m_size];
			for (int i=0;i<m_size;i++)
			{
				m_distance_matrix[i]= new int[m_size];
				m_flow_matrix[i] = new int[m_size];
			}
		}
		else if (1<=num && num<=m_size)
		{
			//LOAD DISTANCE MATRIX
			char * pch;
			pch = strtok (line," ");
			int distance=atoi(pch);
			m_distance_matrix[num-1][0]=distance;
			for (int i=1;i < m_size; i++)
			{
				pch = strtok (NULL, " ,.");
				distance=atoi(pch);
				m_distance_matrix[num-1][i]=distance;
			}
		}
		else if (num>m_size && num<=(2*m_size))
		{
			//LOAD FLOW MATRIX
			char * pch;
			pch = strtok (line," ");
			int weight=atoi(pch);
			m_flow_matrix[num-m_size-1][0]=weight;
			for (int i=1;i < m_size; i++)
			{
				pch = strtok (NULL, " ,.");
				weight=atoi(pch);
				m_flow_matrix[num-m_size-1][i]=weight;
			}
		}
		else
		{
			break;
		}

		num++;
	}

	indata.close();

    //CalculateAverageComponents_ELD();
	return (m_size);
}

/*
 * This function evaluates the individuals for the QAP problem.
 */
double QAP::Evaluate(int * genes)
{
	double fitness=0;
	int FactA, FactB;
	int distAB, flowAB, i ,j;
	for (i=0;i<m_size;i++)
	{
		for (j=0;j<m_size;j++)
		{
			FactA = genes[i];
			FactB = genes[j];
			
			distAB= m_distance_matrix[i][j];
			flowAB= m_flow_matrix[FactA][FactB];
			fitness= fitness+(distAB*flowAB);			
		}
	}
	return fitness;
}

/*
 * Returns the size of the problem.
 */
int QAP::GetProblemSize()
{
    return m_size;
}

/*
 * Omega 1.
 */
inline int Omega1(int i, int j, int p, int q, int * x, int n){
    if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
        return -1; //rho
    if ((x[i]==p)^(x[j]==q))
        return -2; //gamma
    if ((x[i]==q)^(x[j]==p))
        return 0; //epsilon
    if ((x[i]==p)&&(x[j]==q))
        return n-3; //alpha.
    if ((x[i]==q)&&(x[j]==p))
        return 1-n; //beta
    cout<<"Error in Omega 1. "<<endl; exit(1);
}

/*
 * Omega 2.
 */
inline int Omega2(int i, int j, int p, int q, int * x, int n){
    if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
        return 1; //rho
    if ((x[i]==p)^(x[j]==q))
        return 0; //gamma
    if ((x[i]==q)^(x[j]==p))
        return 0; //epsilon
    if ((x[i]==p)&&(x[j]==q))
        return n-3; //alpha.
    if ((x[i]==q)&&(x[j]==p))
        return n-3; //beta
    cout<<"Error in Omega 2. "<<endl; exit(1);
}

/*
 * Omega 2.
 */
inline int Omega3(int i, int j, int p, int q, int * x, int n){
    if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
        return -1; //rho
    if ((x[i]==p)^(x[j]==q))
        return n-2; //gamma
    if ((x[i]==q)^(x[j]==p))
        return 0; //epsilon
    if ((x[i]==p)&&(x[j]==q))
        return 2*n-3; //alpha.
    if ((x[i]==q)&&(x[j]==p))
        return 1; //beta
    cout<<"Error in Omega 3. "<<endl; exit(1);
}

/*
 * Calculates the fitness value corresponding to the first component of the elementary landscape decomposition.
 */
double QAP::fc1(int * x){
    int i,j,p,q;
    double result=0;
    int n=m_size;
    double psi;
    for (i=0;i<n;i++){
        for (j=0;j<n;j++){
            for (p=0;p<n;p++){
                for (q=0;q<n;q++){
                    for (q=0;q<p;q++){
                        psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                        if ((x[i]==p)^(x[j]==q))
                            result+= -2* psi; //gamma Omega1
                        else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                            result-= psi; //zeta Omega1
                        else if (x[i]==p && x[j]==q)
                            result+=(n-3)* psi; //alpha Omega1
                        else if (x[i]==q && x[j]==p)
                            result+=(1-n)*psi; //beta Omega1
                            
                    }
                    
                    for (q=p+1;q<n;q++){
                        psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                        if ((x[i]==p)^(x[j]==q))
                            result+= -2* psi; //gamma Omega1
                        else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                            result-= psi; //zeta Omega1
                        else if (x[i]==p && x[j]==q)
                            result+=(n-3)* psi; //alpha Omega1
                        else if (x[i]==q && x[j]==p)
                            result+=(1-n)*psi; //beta Omega1
                    }

                }
            }
        }
    }
    result=result/(double)(2*m_size);
    return result;
}

/*
 * Calculates the fitness value corresponding to the second component of the elementary landscape decomposition.
 */
double QAP::fc2(int * x){
    int i,j,p,q;
    double result=0;
    int n3=m_size-3;

    float psi;
    for (i=0;i<m_size;i++){
        for (j=0;j<m_size;j++){
            for (p=0;p<m_size;p++){
                for (q=0;q<p;q++){
                    psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                    
                    if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                        result+= psi; //zeta Omega2
                    
                    else if (x[i]==p && x[j]==q)
                        result+=n3* psi; //alpha Omega2
                    
                    else if (x[i]==q && x[j]==p)
                        result+=n3* psi; //beta Omega2
                }
                
                for (q=p+1;q<m_size;q++){
                    psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                    if  (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                        result+= psi; //zeta Omega2
            
                    else if (x[i]==p && x[j]==q)
                        result+=n3* psi; //alpha Omega2
                    
                   else if (x[i]==q && x[j]==p)
                        result+=n3* psi; //beta Omega2
                }
            }

        }
    }
    result=result/(double)(2*(m_size-2));
    return result;
}

/*
 * Calculates the fitness value corresponding to the second component of the elementary landscape decomposition. This method is optimized for computation.
 */
double QAP::fc2_optimized(int * x){
    int i,j,p,q;
    long int result=0;
    long int aux=0;
    int n3=m_size-3;
    int min,max,min_plus,max_plus;
    int * row;
    for (i=0;i<m_size;i++){
        for (j=i+1;j<m_size;j++){
            
            min=MIN(x[i],x[j]);
            max=MAX(x[i],x[j]);
            min_plus=min+1;
            max_plus=max+1;
            aux=0;
            
            //zeta cases
            //there are three intervals, from [0, min-1], [min+1,max-1] and [max+1,n-1];
            //and note that p and q cannot be equal.
            for (p=0;p<min;p++){
                row=m_flow_matrix[p];
                for (q=0;q<p;q++){
                    aux+=row[q];
                }
                for (q=p+1;q<min;q++){
                    aux+=row[q];
                }
                for (q=min_plus;q<max;q++){
                    aux+=row[q];
                }
                for (q=max_plus;q<m_size;q++){
                    aux+=row[q];
                }
            }
            
            for (p=min_plus;p<max;p++){
                row=m_flow_matrix[p];
                for (q=0;q<min;q++){
                    aux+=row[q];
                }
                for (q=min_plus;q<p;q++){
                    aux+=row[q];
                }
                for (q=p+1;q<max;q++){
                    aux+=row[q];
                }
                for (q=max_plus;q<m_size;q++){
                    aux+=row[q];
                }
            }
            
            for (p=max_plus;p<m_size;p++){
                row=m_flow_matrix[p];
                for (q=0;q<min;q++){
                   aux+=row[q];
                }
                for (q=min_plus;q<max;q++){
                    aux+=row[q];
                }
                for (q=max_plus;q<p;q++){
                    aux+=row[q];
                }
                for (q=p+1;q<m_size;q++){
                    aux+=row[q];
                }
            }
            
            //alpha case
            aux+=n3*m_flow_matrix[x[i]][x[j]];
            
            //beta case.
            aux+=n3*m_flow_matrix[x[j]][x[i]];
            
            //we multiply with distance here because the distance is a common factor in the calculation.
            result+=aux*m_distance_matrix[i][j];
        }
    }
    //result=result/(double)(2*(m_size-2));
    return (double)result/(double)(m_size-2);
}

/*
 * Calculates the fitness value corresponding to the second component of the elementary landscape decomposition. This method is optimized for a more efficient computation.
 */
double QAP::fc2_optimizedV2(int * x){
    int i,j,p,q;
    long int result=0;
    long int aux;
    int n3=m_size-3;
    int min,max;//,min_plus,max_plus;
    for (i=0;i<m_size;i++){
        for (j=i+1;j<m_size;j++){
            
            min=MIN(x[i],x[j]);
            max=MAX(x[i],x[j]);
            //min_plus=min+1;
            //max_plus=max+1;
            aux=m_totalpq;
            
            //p==min denenan,
            p=min;
            for (q=0;q<min;q++)
                aux -= m_flow_matrix[p][q];
            for (q=min+1;q<m_size;q++)
                aux -= m_flow_matrix[p][q];
            
            //p==max denenan,
            p=max;
            for (q=0;q<max;q++)
                aux -= m_flow_matrix[p][q];
            for (q=max+1;q<m_size;q++)
                aux -= m_flow_matrix[p][q];
            
            //eta q-rekin berdin:
            
            //q==min denenan,
            q=min;
            for (p=0;p<min;p++)
                aux -= m_flow_matrix[p][q];
            for (p=min+1;p<max;p++)		//--> p==max aurreko pausuan kendu dugu
                aux -= m_flow_matrix[p][q];
            for (p=max+1;p<m_size;p++)
                aux -= m_flow_matrix[p][q];
            
            //q==max denenan,
            q=max;
            for (p=0;p<min;p++)
                aux -= m_flow_matrix[p][q];
            for (p=min+1;p<max;p++)		//--> p==max aurreko pausuan kendu dugu
                aux -= m_flow_matrix[p][q];
            for (p=max+1;p<m_size;p++)
                aux -= m_flow_matrix[p][q];
            
            //alpha case
            aux+=n3*m_flow_matrix[x[i]][x[j]];
            
            //beta case.
            aux+=n3*m_flow_matrix[x[j]][x[i]];
            
            //we multiply with distance here because the distance is a common factor in the calculation.
            result+=aux*m_distance_matrix[i][j];
        }
    }
    //result=result/(double)(2*(m_size-2));
    return (double)result/(double)(m_size-2);
}


/*
 * Calculates the fitness value corresponding to the third component of the elementary landscape decomposition.
 */
double QAP::fc3(int * x){
    int i,j,p,q;
    double result=0;
    int n=m_size;
    double psi;
    for (i=0;i<m_size;i++){
        for (j=0;j<m_size;j++){
            for (p=0;p<m_size;p++){
                for (q=0;q<p;q++){
                    psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                    if ((x[i]==p)^(x[j]==q))
                        result+= (n-2)* psi; //gamma Omega3
                    else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                        result-= psi; //zeta Omega3
                    else if ((x[i]==p)&&(x[j]==q))
                        result+=(2*n-3)*psi; //alpha Omega3
                    else if ((x[i]==q)&&(x[j]==p))
                        result+= psi; //beta Omega3
                }
                
                for (q=p+1;q<m_size;q++){
                    psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                    if ((x[i]==p)^(x[j]==q))
                        result+= (n-2)* psi; //gamma Omega3
                    else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q)
                        result-= psi; //zeta Omega3
                    else if ((x[i]==p)&&(x[j]==q))
                        result+=(2*n-3)*psi; //alpha Omega3
                    else if ((x[i]==q)&&(x[j]==p))
                        result+= psi; //beta Omega3
                }
            }
        }
    }
    result=result/(double)(m_size*(m_size-2));
    
    return result;
}


/*
 * Calculates the fitness value decomposed in the three components of the elementary landscape decomposition.
 */
void QAP::f_components(int * x, double * components){
    int i,j,p,q;
    int n=m_size;
    double result1=0,result2=0,result3=0;
    double psi;
    
    for (i=0;i<m_size;i++){
        for (j=0;j<m_size;j++){
            for (p=0;p<m_size;p++){
                for (q=0;q<p;q++){
                        psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                        if ((x[i]==p)^(x[j]==q)){
                           // result1-=2*psi; //gamma Omega1
                            //result2+=(0)*psi; //gamma Omega2
                            result3+= (n-2)* psi; //gamma Omega3
                        }
                        else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q){
                        //    result1-= psi; //zeta Omega1
                            result2+= psi; //zeta Omega2
                            result3-= psi; //zeta Omega3
                        }
                        //else if ((x[i]==q)^(x[j]==p))
                        //  result1+=0; //epsilon is always 0
                        //  result2+=0; //epsilon is always 0
                        //  result3+=0; //epsilon is always 0
                        else if ((x[i]==p)&&(x[j]==q)){
                       //     result1+=(n-3)*psi; //alpha Omega1
                            result2+=(n-3)*psi; //alpha Omega2
                            result3+=(2*n-3)*psi; //alpha Omega3
                        }
                        else if ((x[i]==q)&&(x[j]==p)){
                        //    result1+=(1-n)*psi; //beta Omega1
                            result2+=(n-3)*psi; //beta Omega2
                            result3+= 1*psi; //beta Omega3
                        }
                }
                for (q=p+1;q<m_size;q++){
                    psi=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                    if ((x[i]==p)^(x[j]==q)){
                    //    result1-=2*psi; //gamma Omega1
                        //result2+=(0)*psi; //gamma Omega2
                        result3+= (n-2)* psi; //gamma Omega3
                    }
                    else if (x[i]!=p && x[i]!=q && x[j]!=p && x[j]!=q){
                    //    result1-= psi; //zeta Omega1
                        result2+= psi; //zeta Omega2
                        result3-= psi; //zeta Omega3
                    }
                    //else if ((x[i]==q)^(x[j]==p))
                    //  result1+=0; //epsilon is always 0
                    //  result2+=0; //epsilon is always 0
                    //  result3+=0; //epsilon is always 0
                    else if ((x[i]==p)&&(x[j]==q)){
                   //     result1+=(n-3)*psi; //alpha Omega1
                        result2+=(n-3)*psi; //alpha Omega2
                        result3+=(2*n-3)*psi; //alpha Omega3
                    }
                    else if ((x[i]==q)&&(x[j]==p)){
                    //    result1+=(1-n)*psi; //beta Omega1
                        result2+=(n-3)*psi; //beta Omega2
                        result3+= 1*psi; //beta Omega3
                    }
                }
                
                
            }
        }
    }

    //components[0]=result1/(double)(2*n);
    components[0]=fc1_avg; //when QAP

    components[1]=result2/(double)(2*(n-2));
    components[2]=result3/(double)(n*(n-2));
 
}
/*
 * Calculates the average fitness components of the elementary landscape decomposition needed to calculate the average
 * fitness of a neighborhood in close form.
 */
void QAP::CalculateAverageComponents_ELD(){
    double phi=0;
    int i,j,p,q;
    for (i=0;i<m_size;i++){
        for (j=0;j<m_size;j++){
            for (p=0;p<m_size;p++){
                for (q=0;q<m_size;q++){
                    if (i!=j && p!=q)
                        phi+=m_distance_matrix[i][j]*m_flow_matrix[p][q];
                }
            }
        }
    }
    fc1_avg= - phi/(2*m_size);
    fc2_avg= phi * (m_size-3)/(2*(m_size-2)*(m_size-1));
    fc3_avg= phi/(m_size*(m_size-2));
    
    m_totalpq=0;
    for (p=0;p<m_size;p++){
        for (q=0;q<p;q++)
            m_totalpq += m_flow_matrix[p][q];
        for (q=p+1;q<m_size;q++)
            m_totalpq += m_flow_matrix[p][q];
    }
}

