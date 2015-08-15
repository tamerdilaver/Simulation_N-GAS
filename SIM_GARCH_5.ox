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
static decl dOMEGA_ERROR;
static decl dGAMMA_ERROR;
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
**  Input:		dAlpha, dBeta, avReturns, iIteration [to get different Zt's]
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
**  Function:	Transform (start)parameters	  Alpha, Beta Startingvalues
**
**  Input: 		vTheta [parametervalues]
**
**  Output: 	vThetaStar
*/

fTransform(const avThetaStar, const vTheta){
	avThetaStar[0] = vTheta;

	avThetaStar[0][0] = log(vTheta[0]);
	avThetaStar[0][1] = log(vTheta[1]);
//	avThetaStar[0][2] = log(vTheta[2]);
//	avThetaStar[0][3] = log(vTheta[3]);
	return 1;
}

/*
**  Function: 	Extract the parameters from vTheta
**
**  Input: 		adAlpha, adBeta, aOmega, adGamma,, vTheta
**
**  Output: 	1 
*/

fGetPars(const adAlpha, const adBeta, const vTheta){

	adAlpha[0] = exp(vTheta[0]);
	adBeta[0] = exp(vTheta[1]);
//	adOmega[0] = exp(vTheta[2]);
//	adGamma[0] = exp(vTheta[3]);
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
	decl dAlpha, dBeta;
	fGetPars( &dAlpha,  &dBeta, vTheta);

	decl dS2 = dGAMMA_ERROR;	//initial condition by definition
	decl vLogEta = zeros(sizerc(s_vY), 1);

	for(decl i = 0; i < sizerc(s_vY); ++i){
			//likelihood contribution
			vLogEta[i] = log(M_2PI) +log(dS2) + s_vY[i]^2 / dS2; //Gaussian
						
			//GARCH recursion
			dS2 = dOMEGA_ERROR + dBeta* dS2 +  dAlpha* s_vY[i]^2;
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
//	avTheta[0][2] = exp(vThetaStar[2]);
//	avTheta[0][3] = exp(vThetaStar[3]);
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

fEstimateGarch(const vReturns, const adAlpha_hat, const adBeta_hat){

	//initialise parameter values
	decl vTheta = zeros(iPARS,1);
	vTheta = <0.1 ; 0.9>; // Alpha, Beta, Omega, Gamma Startingvalues
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

		bSTD_ERROR = FALSE;
		decl dAlpha_hat, dBeta_hat;
		fEstimateGarch(vReturns, &dAlpha_hat, &dBeta_hat);	 //Omega and Gamma also estimated

		mTemp[i][0] =  (dAlpha_hat-dALPHA);
		mTemp[i][1]	=  (dBeta_hat-dBETA);
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

fMonteCarlo2(const amAlpha, const amBeta, const amAlpha2, const amBeta2){

	decl mTemp, mTempAlpha, mTempBeta;
	decl mTemp2, mTempAlpha2, mTempBeta2;
	mTempAlpha = mTempBeta  = zeros((iSIZE/iSTEPS),3);
	mTempAlpha2 = mTempBeta2 = zeros((iSIZE/iSTEPS),3);
	mTemp = mTemp2 =zeros(iB,iPARS);

	decl iSize = iSIZE;

	for(decl j = 0; j<(iSize/iSTEPS) ; j++){
		iSIZE = ((iSTEPS)*(j+1));
		for(decl i = 0; i<iB ; i++){
			decl vReturns;
			fSimGARCH2(dALPHA, dBETA, dOMEGA, dGAMMA, &vReturns, i);
	
			decl dAlpha_hat, dBeta_hat, vSE;
			vSE = fEstimateGarch(vReturns, &dAlpha_hat, &dBeta_hat);	 //Omega and Gamma also estimated
			
			mTemp[i][0] =  sqrt(iSIZE)*(dAlpha_hat-dALPHA);				//SQRT(T)*(\hat_\alpha_T - \alpha_0) ~ N(0, \SIGMA)
			mTemp[i][1]	=  sqrt(iSIZE)*(dBeta_hat-dBETA);
//			mTemp[i][2]	=  sqrt(iSIZE)*(dOmega_hat-dOMEGA);
//			mTemp[i][3]	=  sqrt(iSIZE)*(dGamma_hat-dGAMMA);
			
			mTemp2[i][0] 	=  (dAlpha_hat-dALPHA)/vSE[0];				//(\hat_\alpha_T - \alpha_0)/SE(\hat_\alpha) ~ N(0, 1)
			mTemp2[i][1]	=  (dBeta_hat-dBETA)/vSE[1];
//			mTemp2[i][2]	=  (dOmega_hat-dOMEGA)/vSE[2];
//			mTemp2[i][3]	=  (dGamma_hat-dGAMMA)/vSE[3];

		}
		// v0.025_quantile, vMean, v0.975_quantile;				We get 95%-intervals
		mTempAlpha[j][0] = quantilec(mTemp[][],0.025)'[0];
		mTempAlpha[j][1] = meanc(mTemp[][])'[0];
		mTempAlpha[j][2] = quantilec(mTemp[][],0.975)'[0];
	
		mTempBeta[j][0] = quantilec(mTemp[][],0.025)'[1];
		mTempBeta[j][1] = meanc(mTemp[][])'[1];
		mTempBeta[j][2] = quantilec(mTemp[][],0.975)'[1];

//		mTempOmega[j][0] = quantilec(mTemp[][],0.025)'[2];
//		mTempOmega[j][1] = meanc(mTemp[][])'[2];
//		mTempOmega[j][2] = quantilec(mTemp[][],0.975)'[2];
//	
//		mTempGamma[j][0] = quantilec(mTemp[][],0.025)'[3];
//		mTempGamma[j][1] = meanc(mTemp[][])'[3];
//		mTempGamma[j][2] = quantilec(mTemp[][],0.975)'[3];

		mTempAlpha2[j][0] = quantilec(mTemp2[][],0.025)'[0];
		mTempAlpha2[j][1] = meanc(mTemp2[][])'[0];	  //deletec()
		mTempAlpha2[j][2] = quantilec(mTemp2[][],0.975)'[0];
	
		mTempBeta2[j][0] = quantilec(mTemp2[][],0.025)'[1];
		mTempBeta2[j][1] = meanc(mTemp2[][])'[1];
		mTempBeta2[j][2] = quantilec(mTemp2[][],0.975)'[1];

//		mTempOmega2[j][0] = quantilec(mTemp2[][],0.025)'[2];
//		mTempOmega2[j][1] = quantilec(mTemp2[][],0.5)'[2];
//		mTempOmega2[j][2] = quantilec(mTemp2[][],0.975)'[2];
//	
//		mTempGamma2[j][0] = quantilec(mTemp2[][],0.025)'[3];
//		mTempGamma2[j][1] = quantilec(mTemp2[][],0.5)'[3];
//		mTempGamma2[j][2] = quantilec(mTemp2[][],0.975)'[3];
	}

	amAlpha[0] = mTempAlpha;
	amBeta[0] = mTempBeta;
//	amOmega[0] = mTempOmega;
//	amGamma[0] = mTempGamma;

	amAlpha2[0] = mTempAlpha2;
	amBeta2[0] = mTempBeta2;
//	amOmega2[0] = mTempOmega2;
//	amGamma2[0] = mTempGamma2;

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
	iPARS = 2;



/*
** ..................................................................................	
**	 		Check Bias
**..................................................................................
*/

	//SET # OF SIMULATIONS 
	iB = 100; 			//max 5000
	iSIZE = 1000;		//max 5000
	iSIMS = iB*iSIZE;
	vSTD_NORM = rann(iSIMS,1);
	bSTD_ERROR = TRUE;				 //boolean

	//GET BETA SUCH THAT WE GET EQUALITY "Elog(alpha_0 z_t^2 + beta_0) = 0".
	decl dBeta_0;
	fGetBeta(iSIMS,  &dBeta_0);
	dBETA = dBeta_0;

	
	decl dOmega_min, dOmega_max, dOmega_step, dGamma_min, dGamma_max, dGamma_step;
	dOmega_min = max(0.001,dOMEGA*0.1);
	dOmega_max = dOMEGA*2;
	dOmega_step = dOMEGA/10;

	dGamma_min = max(0.01,dGAMMA*0.1);
	//print(dGamma_min,"\n");
	dGamma_max = dGAMMA*2;
	//print(dGamma_max,"\n");
	dGamma_step = dGAMMA/10;
	//print(dGamma_step,"\n");
	
//	decl vReturns;
//	fSimGARCH(dALPHA, dBETA, dOMEGA, dGAMMA, &vReturns, 0);

	decl vX, vY, mZ, mZ_2;
	
	vX = zeros((dOmega_max-dOmega_min)/dOmega_step,1); //vX are sigma2_eta
	vY = zeros((dGamma_max-dGamma_min)/dGamma_step,1);

//	print(sizerc(vX),"\n");
//	print(sizerc(vY),"\n");
	
	mZ = zeros(sizerc(vX),sizerc(vY));
	mZ_2 = zeros(sizerc(vX),sizerc(vY));
	
	for(decl i = 0; i<sizerc(vX);i++){
		vX[i]  = (i+1)*dOmega_step;
		
		for(decl w = 0; w<sizerc(vY) ; w++){
			vY[w] = (w+1)*dGamma_step;
			dOMEGA_ERROR = vX[i];
			dGAMMA_ERROR = vY[w];

			decl dAlpha_hat, dBeta_hat, vSE;

			//DO MANY SIMULATIONS AND ESITMATIONS	
			decl mMonteCarlo;
			fMonteCarlo(&mMonteCarlo);
			
			mZ[i][w] 	= meanc(mMonteCarlo[][0]);
			mZ_2[i][w] 	= meanc(mMonteCarlo[][1]);
			//print(i~w~mZ[i][w]~mZ_2[i][w]);
		}
		 
	}

	decl iMaxCorrect = maxc(maxc(-fabs(mZ))');
	decl iMaxCorrect2 = maxc(maxc(-fabs(mZ_2))');
	


	print("\nMin estimate error for A: ", iMaxCorrect);
	//print("\nWith $\hat\omega$ = ",vX[maxcindex(maxc( -mZ' )')], " $\hat\gamma$ = ", vY[maxcindex(maxc(-mZ)')]);

	print("\nOptimale parameters met standaarderrors \n",
          	"%r", { "omega",  "gamma"},
          	"%c", {"thetaStart","theta"}, (dOMEGA|dGAMMA)~(vX[maxcindex(maxc( -fabs(mZ)' )')]|vY[maxcindex(maxc(-fabs(mZ))')]));

		print("\nMin estimate error for B: ", iMaxCorrect2);
	//print("\nWith $\hat\omega$ = ",vX[maxcindex(maxc( -mZ' )')], " $\hat\gamma$ = ", vY[maxcindex(maxc(-mZ)')]);

	print("\nOptimale parameters met standaarderrors \n",
          	"%r", { "omega",  "gamma"},
          	"%c", {"thetaStart","theta"}, (dOMEGA|dGAMMA)~(vX[maxcindex(maxc( -fabs(mZ_2)' )')]|vY[maxcindex(maxc(-fabs(mZ_2))')]));

	SetDrawWindow("SIM_GARCH_5(test)");
	DrawXYZ( 0,vY, vX, mZ,0 , 	"$\gamma$", "$\omega$", "$E(\hat A-A_0)$");
	DrawXYZ( 1,vY, vX, mZ_2,0 , "$\gamma$", "$\omega$", "$E(\hat B-B_0)$");
	DrawTitle(0,"(i) $E(\hat A-A_0)$ for fixed $\omega$ and $\gamma$");
	DrawTitle(1,"(ii) $E(\hat B-B_0)$ for fixed $\omega$ and $\gamma$");
	ShowDrawWindow();


}

