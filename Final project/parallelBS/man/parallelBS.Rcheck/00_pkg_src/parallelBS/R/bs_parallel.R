library(parallel)
library(Rcpp, lib.loc="~/R/")

#serial version
basis = function(x, degree, i , knots) {
	if(degree == 0) 
			B = ifelse((x >= knots[i]) & (x < knots[i+1]), 1, 0)
	else {
		#alpha1 computation
		if((knots[degree+i] - knots[i]) == 0) 
				alpha1 = 0
		else
				alpha1 = (x - knots[i])/(knots[degree+i] - knots[i])
		#alpha2 computation
		if((knots[i+degree+1] - knots[i+1]) == 0)
				alpha2 = 0
		else
				alpha2 = (knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1])
		B = alpha1*basis(x, (degree-1), i, knots) + alpha2*basis(x, (degree-1), (i+1), knots)
	}
	return(B)
}


#snow version, clusters required
formMatrixSnow = function(cls, x, degree, knots, K){
    sequence = 1:K

    #returns column of the matrix
    basis = function(i,degree){
        if(degree == 0){
            B = ifelse((x >= knots[i]) & (x < knots[i+1]), 1, 0)
        }
        else {
            #alpha1 computation
            if((knots[degree+i] - knots[i]) == 0)
                alpha1 = 0
            else
                alpha1 = (x - knots[i])/(knots[degree+i] - knots[i])
            #alpha2 computation
            if((knots[i+degree+1] - knots[i+1]) == 0)
                alpha2 = 0
            else
                alpha2 = (knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1])           
            B = alpha1*basis(i,(degree-1)) + alpha2*basis((i+1),(degree-1))
        }
        return(B)
    }
    
    #have each cluster obtain a column of the matrix
    jvector = clusterApply(cls,sequence,basis,degree)
    #form the final matrix to return
    Bmat = Reduce(cbind,jvector)    
}

#bs function call
bs = function(x, degree=3, interior.knots=NULL, intercept=FALSE, Boundary.knots = c(0,1), type="serial", ncls=2) {

    	#input check
    	if(missing(x)) 
        stop("You must provide x")
    	if(degree < 1) 
        stop("The spline degree must be at least 1")
                                                
      #input preprocessing
    	Boundary.knots = sort(Boundary.knots)
    	interior.knots.sorted = NULL
    	if(!is.null(interior.knots)) 
        	interior.knots.sorted = sort(interior.knots)
                                                                                    
    	#formation of knots
    	knots <- c(rep(Boundary.knots[1], (degree+1)), interior.knots.sorted, rep(Boundary.knots[2], (degree+1)))
    	K <- length(interior.knots) + degree + 1
	  	lenx = length(x)

    	#parallel calls
			if(type == "openmp") {
				print("run type: openmp")
				dyn.load("bs_omp.so")
				Bmat = .Call("formMatrix", x, degree, knots, K, lenx)
			} 
			else if (type=="cuda") {
				print("run type: cuda")
				dyn.load("bs_cuda.so")
				Bmat = .Call("formMatrix", x, degree, knots, K, lenx)
			}
			else if(type=="snow") {
				print("run type: snow")
				cls = makePSOCKcluster(rep("localhost", ncls))
				print(cls)
				Bmat = formMatrixSnow(cls, x, degree, knots, K)
			}
			else if(type=="serial") {
				print("run type: serial")
				Bmat = matrix(0,length(x),K)
				for(j in 1:K)
						Bmat[,j] = basis(x, degree, j, knots)	
			} 
			else {
				print("Incorrect run type - serial used instead")
				#do serial
				Bmat = matrix(0,length(x),K)
				for(j in 1:K) 
							Bmat[,j] = basis(x, degree, j, knots)				
			}
    		
      #output postprocessing
      if(any(x == Boundary.knots[2])) 
          Bmat[x == Boundary.knots[2], K] <- 1
      if(intercept == FALSE)
          return(Bmat[,-1])
      else
            return(Bmat)

} #end bs function


#testing stuff
#n <- 100
#x <- seq(0, 1, length=n)

#c2 <- makePSOCKcluster(rep("localhost",2))

#(B <- bs(c2, x, degree=5, intercept = TRUE, Boundary.knots=c(0,1)))

#B <- bs(c2, x, degree=5, intercept = TRUE, Boundary.knots=c(0,1))
#(B <- bs(x, degree=5, intercept = TRUE, Boundary.knots=c(0,1),type="serial",ncls=4))

