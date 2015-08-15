/*
**	Master Thesis Econometrics
**
**  Purpose:
**  	For some fixed parameter values simulate and estimate GARCH model parameters
**		with Maximum Likelikhood many times. s.t. Elog(alpha_0 z_t^2 + beta_0) = 0.
**
**  Date:
**    	30/05/2015
**
**  Author:
**	  	Tamer Dilaver
**
**	Supervisor:
**		Fransisco Blasques
**
*/


/*		 [[[[[[DIT MOET WEG!!!]]]]]]
**	Case Study Financial Econometrics 4.3 
**
**  Purpose:
**  	Estimate all GARCH model parameters (gamma, omega, alpha and beta)
**		with Maximum Likelikhood many times. s.t. Elog(alpha_0 z_t^2 + beta_0) < 0. (or simply alpha + beta <1) (Since alpha>0 and beta>0) 
**
**  Date:
**    	10/01/2015
**
**  Author:
**	  	Tamer Dilaver, Koen de Man & Sina Zolnoor
**
**	Supervisor:
**		L.F. Hoogerheide & S.J. Koopman
**
*/

#include <oxstd.h>
#include <oxdraw.h>
#include <oxprob.h>
#include <maximize.h>
#import <modelbase>
#import <simula>
#include <oxfloat.h>

static decl iB;	 					//Repeats
static decl iSIZE;					//Size of time series
static decl iSTEPS;					//#Steps to divide the size
static decl iSIMS;					//# of Zt ~ N(0,1)
static decl dALPHA;
static decl dBETA;
static decl dOMEGA;
static decl dGAMMA;
static decl iPARS;					//number of parameters
static decl vSTD_NORM;				// Zt ~ N(0,1)
static decl mSTD_NORM;
static decl s_vY; 					//Simulated returns
static decl	bSTD_ERROR;				//0 or 1 boalean


/*
**  Function:	Transform (start)parameters
**
**  Input: 		vTheta [parametervalues]
**
**  Output: 	vThetaStar
*/
fTransform2(const avThetaStar, const vTheta){
	avThetaStar[0]=		vTheta;
	
	avThetaStar[0][0] = log(vTheta[0]);

	return 1;
}

/*
**  Function:	Transform parameters back
**
**  Input: 		vThetaStar
**
**  Output: 	vTheta [parametervalues]
*/
fTransformBack2(const avTheta, const vThetaStar){
	avTheta[0]=		vThetaStar;

	avTheta[0][0] = exp(vThetaStar[0]);
	return 1;
}

/*
**  Function: 	Extract the parameters from vTheta
**
**  Input: 		adBeta, vTheta
**
**  Output: 	1 
*/
fGetPars2(const adBeta, const vTheta){

	adBeta[0] = exp(vTheta[0]);
	return 1;
}

/*
**  Function:	Calculate targetvalue -[ E log(alpha_0 z_t^2 + beta_0) ]^2 for given parameters
**
**  Input:		vTheta [parametervalues], adFunc [adres functionvalue], avScore [the score],  amHessian [hessianmatrix]
**
**  Output:		1
**
*/
fExpectation(const vTheta, const adFunc, const avScore, const amHessian){
	decl dBeta;
	fGetPars2( &dBeta, vTheta);

	//NOTICE: Since maxBFGS() is a function that maximimises I have squared the target function
	//and then I have put a negative sign in front of it. This ensures that maximising will
	//effectively search for the value for when the target function is zero.
	adFunc[0] = - (meanc(log(fabs(dALPHA * vSTD_NORM.^2 + dBeta))))^2; 						
	return 1;
}

/*
**  Function:	Get the value for beta subject to Elog(alpha_0 z_t^2 + beta_0) = 0 for given parameters	and simulated Zt's
**
**  Input:		iSims, adBeta  
**
**  Output:		1
**
*/
fGetBeta(const iSims, const adBeta)
{
	decl vTheta, vThetaStart, vThetaStar, dFunc, iA;

	//initialise startparameter(s)
	vTheta = zeros(1,1);
	vTheta = <0.9>;			// dBeta
	vThetaStart = vTheta;

	//transform startparameter(s)
	fTransform2(&vThetaStar, vTheta);

	//maximise
	iA=MaxBFGS(fExpectation, &vThetaStar, &dFunc, 0, TRUE);
	
	//Transform thetasStar back
   	fTransformBack2(&vTheta, vThetaStar);

	print("\nStart & Optimal parameter(s) with alpha fixed at ",dALPHA," and ",iSIMS," simulations such that we get I(1). \n",
          "%r", { "dBeta"},
          "%c", {"thetaStart","theta"}, vThetaStart~vTheta);
	adBeta[0] = vTheta[0];

	return 1;
}

/*
**  Function:	Simulate GARCH returns for given parameters
**
**  Input:		dAlpha, dBeta, dOmega, dGamma, avReturns, iIteration [to get different Zt's]
**
**  Output:		1
**
*/

fSimGARCH(const dAlpha, const dBeta, const dOmega, const dGamma, const avReturns, const iIteration){
	decl vTemp, vH;
	vTemp = vH =  zeros(iSIZE+1, 1);

	vH[0]= dGamma;		//by definition
	
	for(decl i = 0; i < iSIZE; i++){	
		vTemp[i] =  sqrt(vH[i])*vSTD_NORM[(i + (iIteration*iSIZE))];
		vH[i+1] = dOmega+ dBeta*vH[i] + dAlpha*sqr(vTemp[i]) ;
	}

	vTemp = dropr(vTemp,iSIZE);
	vH = dropr(vH,iSIZE);
	
	avReturns[0] = vTemp;
	return 1;
}

fSimGARCH2(const dAlpha, const dBeta, const dOmega, const dGamma, const avReturns, const iIteration){
	decl vTemp, vH;
	vTemp = vH =  zeros(iSIZE+1, 1);

	vH[0]= dGamma;		//by definition
	
	for(decl i = 0; i < iSIZE; i++){	
		vTemp[i] =  sqrt(vH[i])*mSTD_NORM[iIteration][i];
		vH[i+1] = dOmega+ dBeta*vH[i] + dAlpha*sqr(vTemp[i]) ;
	}

	vTemp = dropr(vTemp,iSIZE);
	vH = dropr(vH,iSIZE);
	
	avReturns[0] = vTemp;
	return 1;
}

/*
**  Function:	Transform (start)parameters	  Alpha, Beta, Omega, Gamma Startingvalues
**
**  Input: 		vTheta [parametervalues]
**
**  Output: 	vThetaStar
*/

fTransform(const avThetaStar, const vTheta){
	avThetaStar[0] = vTheta;

	avThetaStar[0][0] = log(vTheta[0]);
	avThetaStar[0][1] = log(vTheta[1]);
	avThetaStar[0][2] = log(vTheta[2]);
	avThetaStar[0][3] = log(vTheta[3]);
	return 1;
}

/*
**  Function: 	Extract the parameters from vTheta
**
**  Input: 		adAlpha, adBeta, aOmega, adGamma,, vTheta
**
**  Output: 	1 
*/

fGetPars(const adAlpha, const adBeta, const adOmega, const adGamma, const vTheta){

	adAlpha[0] = exp(vTheta[0]);
	adBeta[0] = exp(vTheta[1]);
	adOmega[0] = exp(vTheta[2]);
	adGamma[0] = exp(vTheta[3]);
	return 1;
}

/*
**  Function:	Calculates average value loglikelihood for GARCH given parameter values
**
**  Input: 		vTheta [parameter values], adFunc [adress functievalue], avScore [the score], amHessian [hessian matrix]
**
**  Output:		1
**
*/

fLogLike_Garch(const vTheta, const adFunc, const avScore, const amHessian){
	decl dAlpha, dBeta, dOmega, dGamma;
	fGetPars( &dAlpha,  &dBeta, &dOmega,  &dGamma, vTheta);

	decl dS2 = dGamma;	//initial condition by definition
	decl vLogEta = zeros(sizerc(s_vY), 1);

	for(decl i = 0; i < sizerc(s_vY); ++i){
			//likelihood contribution
			vLogEta[i] = log(M_2PI) +log(dS2) + s_vY[i]^2 / dS2; //Gaussian
						
			//GARCH recursion
			dS2 = dOmega + dBeta* dS2 +  dAlpha* s_vY[i]^2;
	}
	
	adFunc[0] = sumc(vLogEta)/(-2*sizerc(s_vY)); //Average
	return 1;
}

/*
**  Function:	Transform parameters back	Alpha, Beta, Omega, Gamma Startingvalues
**
**  Input: 		vThetaStar
**
**  Output: 	vTheta [parametervalues]
*/

fTransformBack(const avTheta, const vThetaStar){
	avTheta[0]=		vThetaStar;

	avTheta[0][0] = exp(vThetaStar[0]);
	avTheta[0][1] = exp(vThetaStar[1]);
	avTheta[0][2] = exp(vThetaStar[2]);
	avTheta[0][3] = exp(vThetaStar[3]);
	return 1;
}

/*
**  Function:	calculate standard errors
**
**  Input: 		vThetaStar
**
**  Output: 	vStdErrors
*/
fSigmaStdError(const vThetaStar){

 		decl iN, mHessian, mHess, mJacobian, vStdErrors, vP;

		iN 			= sizerc(s_vY);
		Num2Derivative(fLogLike_Garch, vThetaStar, &mHessian);
		//NumJacobian(fTransformBack, vThetaStar, &mJacobian);	  	//numerical Jacobian
		//mHessian 	= mJacobian*invert(-iN*mHessian)*mJacobian';
		mHessian 	= invertgen(-iN*mHessian);
		vStdErrors 	= exp(vThetaStar).*sqrt(diagonal(mHessian)');	//analytisch

		return 	vStdErrors;
}

/*
**  Function:	Estimate Garch parameters
**
**  Input: 		vReturns, adAlpha_hat, adBeta_hat, adOmega_hat, adGamma_hat
**
**  Output: 	vTheta [estimated parametervalues]
*/

fEstimateGarch(const vReturns, const adAlpha_hat, const adBeta_hat, const adOmega_hat, const adGamma_hat){

	//initialise parameter values
	decl vTheta = zeros(iPARS,1);
	vTheta = <0.13 ; 0.998991 ; 0.01 ; 0.1>; // Alpha, Beta, Omega, Gamma Startingvalues
	decl vThetaStart = vTheta;

	//globalalize returns and vectorize true pars
	s_vY = vReturns;

	//transform parameters
	decl vThetaStar; 
	fTransform(&vThetaStar, vTheta);

	//Maximize the LL
	decl dFunc;
	decl iA;
	iA=MaxBFGS(fLogLike_Garch, &vThetaStar, &dFunc, 0, TRUE);

	//Transform thetasStar back
  	fTransformBack(&vTheta, vThetaStar);

	//return alpha, beta, omega and gamma
	adAlpha_hat[0] = vTheta[0];
	adBeta_hat[0] = vTheta[1];
	adOmega_hat[0] = vTheta[2];
	adGamma_hat[0] = vTheta[3];

	if(bSTD_ERROR){		//only do this for fMonteCarlo2
		decl vSigmaStdError = fSigmaStdError(vThetaStar);
		return vSigmaStdError;
	}else{
		return 1;
	}
}

/*
**  Function:	Simulates and Estimates Garch data and parameters many times
**				to illustrate Asymptotic normality
**
**  Input: 		amMonteCarlo [matrix of many estimated parameters];
**
**  Output: 	1
*/

fMonteCarlo(const amMonteCarlo){
	decl mTemp;
	mTemp = zeros(iB,iPARS);

	for(decl i = 0; i<iB ; i++){
		decl vReturns;
		fSimGARCH(dALPHA, dBETA, dOMEGA, dGAMMA, &vReturns, i);

		decl dAlpha_hat, dBeta_hat, dOmega_hat, dGamma_hat, vSE;
		vSE = fEstimateGarch(vReturns, &dAlpha_hat, &dBeta_hat, &dOmega_hat, &dGamma_hat);	 //Omega and Gamma also estimated

		mTemp[i][0]	=  (dAlpha_hat-dALPHA)/vSE[0];	
		mTemp[i][1]	=  (dBeta_hat-dBETA)/vSE[1];
		mTemp[i][2]	=  (dOmega_hat-dOMEGA)/vSE[2];
		mTemp[i][3]	=  (dGamma_hat-dGAMMA)/vSE[3];
	}
	amMonteCarlo[0] = mTemp;
	return 1;
}

/*
**  Function:	Simulated and Estimates Garch data and parameters many times
**				to illustrate consistency it returns minimum, mean and maximum values for the estimated parameters
**
**  Input: 		amAlpha [matrix containing the min, max and mean of estimated alpha],
**				amBeta [matrix containing the min, max and mean of estimated beta], 
**				amOmega [matrix containing the min, max and mean of estimated omega],
**				amGamma [matrix containing the min, max and mean of estimated gamma]
**
**  Output: 	1
*/

fMonteCarlo2(const amAlpha, const amBeta, const amOmega, const amGamma, const amAlpha2, const amBeta2, const amOmega2, const amGamma2){

	decl mTemp, mTempAlpha, mTempBeta, mTempOmega, mTempGamma;
	decl mTemp2, mTempAlpha2, mTempBeta2, mTempOmega2, mTempGamma2;

	decl iPunten;
	iPunten = floor((iSIZE-1000)/iSTEPS);

	mTempAlpha = mTempBeta = mTempOmega = mTempGamma = zeros(iPunten,3);
	mTempAlpha2 = mTempBeta2 = mTempOmega2 = mTempGamma2 = zeros(iPunten,3);
	mTemp = mTemp2 =zeros(iB,iPARS);

	decl iSize = iSIZE;

	
	for(decl j = 0; j<floor((iSize-1000)/iSTEPS) ; j++){
		iSIZE = 1000+iSTEPS*j;
		for(decl i = 0; i<iB ; i++){
			decl vReturns;
			fSimGARCH2(dALPHA, dBETA, dOMEGA, dGAMMA, &vReturns, i);
	
			decl dAlpha_hat, dBeta_hat, dOmega_hat, dGamma_hat, vSE;
			vSE = fEstimateGarch(vReturns, &dAlpha_hat, &dBeta_hat, &dOmega_hat, &dGamma_hat);	 //Omega and Gamma also estimated
			
			mTemp[i][0] =  sqrt(iSIZE)*(dAlpha_hat-dALPHA);				//SQRT(T)*(\hat_\alpha_T - \alpha_0) ~ N(0, \SIGMA)
			mTemp[i][1]	=  sqrt(iSIZE)*(dBeta_hat-dBETA);
			mTemp[i][2]	=  sqrt(iSIZE)*(dOmega_hat-dOMEGA);
			mTemp[i][3]	=  sqrt(iSIZE)*(dGamma_hat-dGAMMA);
			
			mTemp2[i][0] 	=  (dAlpha_hat-dALPHA)/vSE[0];				//(\hat_\alpha_T - \alpha_0)/SE(\hat_\alpha) ~ N(0, 1)
			mTemp2[i][1]	=  (dBeta_hat-dBETA)/vSE[1];
			mTemp2[i][2]	=  (dOmega_hat-dOMEGA)/vSE[2];
			mTemp2[i][3]	=  (dGamma_hat-dGAMMA)/vSE[3];

		}
		// v0.025_quantile, vMean, v0.975_quantile;				We get 95%-intervals
		decl vMeanTemp, vQ0025Temp, vQ0975Temp;
		
		vMeanTemp = meanc(mTemp);
		for(decl i=0;i<sizerc(vecindex(isdotnan(meanc(mTemp))));i++){
			vMeanTemp[vecindex(isdotnan(meanc(mTemp)))[i]] = meanc(deleter(mTemp[:][vecindex(isdotnan(meanc(mTemp)))[i]]));
		}
		vQ0025Temp = quantilec(mTemp,0.025);
		vQ0975Temp = quantilec(mTemp,0.975);

		mTempAlpha[j][0] = vQ0025Temp'[0];
		mTempAlpha[j][1] = vMeanTemp'[0];
		mTempAlpha[j][2] = vQ0975Temp'[0];
	
		mTempBeta[j][0] = vQ0025Temp'[1];
		mTempBeta[j][1] = vMeanTemp'[1];
		mTempBeta[j][2] = vQ0975Temp'[1];

		mTempOmega[j][0] = vQ0025Temp'[2];
		mTempOmega[j][1] = vMeanTemp'[2];
		mTempOmega[j][2] = vQ0975Temp'[2];
	
		mTempGamma[j][0] = vQ0025Temp'[3];
		mTempGamma[j][1] = vMeanTemp'[3];
		mTempGamma[j][2] = vQ0975Temp'[3];

		vMeanTemp = meanc(mTemp2);
		for(decl i=0;i<sizerc(vecindex(isdotnan(meanc(mTemp2))));i++){
			vMeanTemp[vecindex(isdotnan(meanc(mTemp2)))[i]] = meanc(deleter(mTemp2[:][vecindex(isdotnan(meanc(mTemp2)))[i]]));
		}
		vQ0025Temp = quantilec(mTemp2,0.025);
		vQ0975Temp = quantilec(mTemp2,0.975);

		mTempAlpha2[j][0] = vQ0025Temp'[0];
		mTempAlpha2[j][1] = vMeanTemp'[0];	  //deletec()
		mTempAlpha2[j][2] = vQ0975Temp'[0];
	
		mTempBeta2[j][0] = vQ0025Temp'[1];
		mTempBeta2[j][1] = vMeanTemp'[1];
		mTempBeta2[j][2] = vQ0975Temp'[1];

		mTempOmega2[j][0] = vQ0025Temp'[2];
		mTempOmega2[j][1] = vMeanTemp'[2];
		mTempOmega2[j][2] = vQ0975Temp'[2];
	
		mTempGamma2[j][0] = vQ0025Temp'[3];
		mTempGamma2[j][1] = vMeanTemp'[3];
		mTempGamma2[j][2] = vQ0975Temp'[3];
	}

	amAlpha[0] = mTempAlpha;
	amBeta[0] = mTempBeta;
	amOmega[0] = mTempOmega;
	amGamma[0] = mTempGamma;

	amAlpha2[0] = mTempAlpha2;
	amBeta2[0] = mTempBeta2;
	amOmega2[0] = mTempOmega2;
	amGamma2[0] = mTempGamma2;

	return 1;
}

/*
**				MAIN PROGRAM
**
**  Purpose:	Simulate GARCH returns for alpha, omega, gamma and beta many times.
**				Estimate GARCH parameters alpha, beta, omega and gamma.
**
**  Input: 		dALPHA, dBETA, dOMEGA, dGAMMA, iB, iSIZE, iSIMS, iSTEPS
**
**  Output: 	Figures
*/
main()
{
	//SET PARAMETERS
	dALPHA = 0.1;
// 	dBETA = 0.89;
	dOMEGA = 0.01;
	dGAMMA = 0.1;
//	dGAMMA = dOMEGA/(1-dALPHA-dBETA); //doesn't work becomes negative
	iPARS = 4;


/*
** ..................................................................................	
**	 		ASYMPTOTIC NORMALITY
**	Get distributions of alpha and beta (to check for asymptotic normality)
**..................................................................................
*/

	//SET # OF SIMULATIONS 
	iB = 5000; 			//max 5000
	iSIZE = 5000;		//max 5000
	iSIMS = iB*iSIZE;
	vSTD_NORM = rann(iSIMS,1);
	bSTD_ERROR = TRUE;				 //boolean

	//GET BETA SUCH THAT WE GET INEQUALITY "Elog(alpha_0 z_t^2 + beta_0) > 0".
	decl dBeta_0;
	fGetBeta(iSIMS,  &dBeta_0);
	dALPHA = 1.001*dALPHA;
//	dBETA = dBeta_0;
	print("\ndALPHA = ", dALPHA,"\n");
	dBETA = 1.001*dBeta_0;				//100.1% of non-stationarity beta
	print("\ndBeta_0 = ",dBeta_0, " but dBETA = ", dBETA,"\n");
	
	//DO MANY SIMULATIONS AND ESITMATIONS	
	decl mMonteCarlo;
	fMonteCarlo(&mMonteCarlo);	  

	//DRAW GRAPHS
	SetDrawWindow("SIM_GARCH_3");
	DrawDensity(0, (mMonteCarlo[][0])', {"(i) Density $\hat A_1$ AsymN"});
	DrawDensity(1, (mMonteCarlo[][1])', {"(ii) Density $\hat B_1$ AsymN"});
	DrawDensity(2, (mMonteCarlo[][2])', {"(iii) Density $\hat \omega$ AsymN"});
	DrawDensity(3, (mMonteCarlo[][3])', {"(iv) Density $\hat \gamma$ AsymN"});
	DrawDensity(0, vSTD_NORM', {"Z~N(0,1)"});
	DrawDensity(1, vSTD_NORM', {"Z~N(0,1)"});
	DrawDensity(2, vSTD_NORM', {"Z~N(0,1)"});
	DrawDensity(3, vSTD_NORM', {"Z~N(0,1)"});
	DrawTitle(0,"(i) $\hat A_1$ AsymN");	
	DrawTitle(1,"(ii) $\hat B_1$ AsymN");
	DrawTitle(2,"(iii) $\hat \omega$ AsymN");	
	DrawTitle(3,"(iv) $\hat \gamma$ AsymN");
	//ShowDrawWindow();

	print("\nFirst Graph Finished at ",time(),"\n");
/*
** ..................................................................................	
**	 			CONSISTENCY
**	Check consistency for alpha and beta
** ..................................................................................
*/	

	//SET # OF SIMULATIONS 
	iB = 100;			//100
	iSIZE = 100000;		//10000
	iSIMS = iB*iSIZE;
	mSTD_NORM = rann(iB,iSIZE);
	vSTD_NORM = vec(mSTD_NORM);
	bSTD_ERROR = TRUE;

	//GET BETA SUCH THAT WE GET INEQUALITY "Elog(alpha_0 z_t^2 + beta_0) > 0".
	//decl dBeta_0;
	dALPHA = 0.1;
	fGetBeta(iSIMS,  &dBeta_0);
	dBETA = 1.001*dBeta_0;				//100.1% of non-stationarity beta
	print("\ndBeta_0 = ",dBeta_0, " but dBETA = ", dBETA,"\n");
//	dBETA = dBeta_0;
	dALPHA = 1.001*dALPHA;
	print("\ndALPHA = ", dALPHA,"\n");
	
	//DO MANY SIMULATIONS AND ESITMATIONS
	decl mAlpha, mBeta, mOmega, mGamma, mAlpha2, mBeta2, mOmega2, mGamma2;
	iSTEPS = iSIZE/20;				 	//steps of iSIZE/100 takes a while (steps of iSIZE/10 is faster)
	fMonteCarlo2(&mAlpha, &mBeta, &mOmega, &mGamma, &mAlpha2, &mBeta2, &mOmega2, &mGamma2);

	//DRAW GRAPHS
//	SetDrawWindow("SIM_GARCH_2_Cons");
	Draw(4, (mAlpha)',1000,iSTEPS);
	Draw(5, (mBeta)',1000,iSTEPS);
	Draw(6, (mOmega)',1000,iSTEPS);
	Draw(7, (mGamma)',1000,iSTEPS);
	DrawTitle(4,"(v) $\hat A_1$ Cons");	
	DrawTitle(5,"(vi) $\hat B_1$ Cons");
	DrawTitle(6,"(vii) $\hat \omega$ Cons");	
	DrawTitle(7,"(viii) $\hat \gamma$ Cons");
//	ShowDrawWindow();
	print("\nSecond Graph Finished at ",time(),"\n");

//	SetDrawWindow("SIM_GARCH_2_NormC");
	Draw(8, (mAlpha2)',1000,iSTEPS);
	Draw(9, (mBeta2)',1000,iSTEPS);
	Draw(10, (mOmega2)',1000,iSTEPS);
	Draw(11, (mGamma2)',1000,iSTEPS);
	DrawTitle(8,"(ix) $\hat A_1$ NormC");	
	DrawTitle(9,"(x) $\hat B_1$ NormC");
	DrawTitle(10,"(xi) $\hat \omega$ NormC");
	DrawTitle(11,"(xii) $\hat \gamma$ NormC");
	ShowDrawWindow();
	print("\nThird Graph Finished at ",time(),"\n");
}

