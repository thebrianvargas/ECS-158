#rmandel implementation with interface to C++

#Input:
#  nth - number of threads
#  xl  - left limit
#  xr  - right limit
#  yt  - top limit
#  yb  - bot limit
#  inc - distance between ticks
#  maxiters - max number of iterations
#  sched - OMP scheduling method (static,dynamic,guided)
#  chunksize - OMP chunk size
#Output:
#  image ---

#Note OMP scheduling: schedule(type,chunk)
#Still have to put in the time trial code in R

library(Rcpp, lib.loc="~/R/")
dyn.load("p2.so")

rmandel = function(nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize){
   g = list()
   g$x = seq(xl,xr,inc)
   g$y = seq(yb,yt,inc)

   print(
     system.time(
       (g$z = .Call("cmandel",nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize))
      )
   )
   
   image(g, main="Mandelbrot Set", xlab="Real Values", ylab="Imaginary Values")
}

madelopt = function(nth, xl, xr, yb, yt, inc, maxiters) {
  rmandel(nth,xl,xr,yb,yt,inc,maxiters,"static",65536) #run the function
}
