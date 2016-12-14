\name{bs}
\alias{bs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
bs
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
bs(x, degree = 3, interior.knots = NULL, intercept = FALSE, Boundary.knots = c(0, 1), type = "serial", ncls = 2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
%%     ~~Describe \code{x} here~~
}
  \item{degree}{
%%     ~~Describe \code{degree} here~~
}
  \item{interior.knots}{
%%     ~~Describe \code{interior.knots} here~~
}
  \item{intercept}{
%%     ~~Describe \code{intercept} here~~
}
  \item{Boundary.knots}{
%%     ~~Describe \code{Boundary.knots} here~~
}
  \item{type}{
%%     ~~Describe \code{type} here~~
}
  \item{ncls}{
%%     ~~Describe \code{ncls} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (x, degree = 3, interior.knots = NULL, intercept = FALSE, 
    Boundary.knots = c(0, 1), type = "serial", ncls = 2) 
{
    if (missing(x)) 
        stop("You must provide x")
    if (degree < 1) 
        stop("The spline degree must be at least 1")
    Boundary.knots = sort(Boundary.knots)
    interior.knots.sorted = NULL
    if (!is.null(interior.knots)) 
        interior.knots.sorted = sort(interior.knots)
    knots <- c(rep(Boundary.knots[1], (degree + 1)), interior.knots.sorted, 
        rep(Boundary.knots[2], (degree + 1)))
    K <- length(interior.knots) + degree + 1
    lenx = length(x)
    if (type == "openmp") {
        print("run type: openmp")
        dyn.load("bs_omp.so")
        Bmat = .Call("formMatrix", x, degree, knots, K, lenx)
    }
    else if (type == "cuda") {
        print("run type: cuda")
        dyn.load("bs_cuda.so")
        Bmat = .Call("formMatrix", x, degree, knots, K, lenx)
    }
    else if (type == "snow") {
        print("run type: snow")
        cls = makePSOCKcluster(rep("localhost", ncls))
        print(cls)
        Bmat = formMatrixSnow(cls, x, degree, knots, K)
    }
    else if (type == "serial") {
        print("run type: serial")
        Bmat = matrix(0, length(x), K)
        for (j in 1:K) Bmat[, j] = basis(x, degree, j, knots)
    }
    else {
        print("Incorrect run type - serial used instead")
        Bmat = matrix(0, length(x), K)
        for (j in 1:K) Bmat[, j] = basis(x, degree, j, knots)
    }
    if (any(x == Boundary.knots[2])) 
        Bmat[x == Boundary.knots[2], K] <- 1
    if (intercept == FALSE) 
        return(Bmat[, -1])
    else return(Bmat)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
