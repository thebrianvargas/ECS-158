\name{formMatrixSnow}
\alias{formMatrixSnow}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Form Matrix Snow
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
formMatrixSnow(cls, x, degree, knots, K)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{cls}{
%%     ~~Describe \code{cls} here~~
}
  \item{x}{
%%     ~~Describe \code{x} here~~
}
  \item{degree}{
%%     ~~Describe \code{degree} here~~
}
  \item{knots}{
%%     ~~Describe \code{knots} here~~
}
  \item{K}{
%%     ~~Describe \code{K} here~~
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
function (cls, x, degree, knots, K) 
{
    sequence = 1:K
    basis = function(i, degree) {
        if (degree == 0) {
            B = ifelse((x >= knots[i]) & (x < knots[i + 1]), 
                1, 0)
        }
        else {
            if ((knots[degree + i] - knots[i]) == 0) 
                alpha1 = 0
            else alpha1 = (x - knots[i])/(knots[degree + i] - 
                knots[i])
            if ((knots[i + degree + 1] - knots[i + 1]) == 0) 
                alpha2 = 0
            else alpha2 = (knots[i + degree + 1] - x)/(knots[i + 
                degree + 1] - knots[i + 1])
            B = alpha1 * basis(i, (degree - 1)) + alpha2 * basis((i + 
                1), (degree - 1))
        }
        return(B)
    }
    jvector = clusterApply(cls, sequence, basis, degree)
    Bmat = Reduce(cbind, jvector)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
