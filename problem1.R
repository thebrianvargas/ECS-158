#testing values
ms = matrix(c(1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,1), nrow = 4, ncol = 4);
mt = matrix(c(1,1,1,3,4,4,1,2,3,1,2,4), nrow = 6, ncol = 2);
ml = list(c(1, 2, 3), c(NA), c(1), c(2, 4))

squaretolist = function(m) {
  toList = list(1:nrow(m));
  for(row in 1:nrow(m)) {
    edges = c();
    for(col in 1:ncol(m)) {
      if(m[row, col] != 0) edges = c(edges, col);
    }
    if(length(edges) == 0) edges = c(edges, NA);
    toList[[row]] = edges;
  }
  
  return(toList);
}#squaretolist

squaretothin = function(m) {
  toThin = matrix(nrow=0, ncol=2);
  for(row in 1:nrow(m)) {
    for(col in 1:ncol(m)) {
      if(m[row, col] != 0) toThin = rbind(toThin, c(row, col));
    }
  }
  
  return(toThin);
}#squaretothin

thintolist = function(thin, nvert) {
  return(squaretolist(thintosquare(thin, nvert)));
}#thintolist

thintosquare = function(thin, nvert) {
  toSquare = matrix(0, nrow=nvert, ncol=nvert);
  for(row in 1:nrow(thin)) {
    toSquare[thin[row, 1], thin[row, 2]] = 1;
  }
  
  return(toSquare)
}#thintosquare

listtothin = function(inlist) {
  toThin = matrix(nrow=0, ncol=2);
  for(list in 1:length(inlist)) {
    for(item in 1:length(inlist[[list]])) {
      element = inlist[[list]][item];
      if(!is.na(element)) toThin = rbind(toThin, c(list, element));
    }
  }
  
  return(toThin);
}#listtothin

listtosquare = function(inlist, nvert) {
  toSquare = matrix(0, nrow=nvert, ncol=nvert);
  for(list in 1:length(inlist)) {
    for(item in 1:length(inlist[[list]])) {
      element = inlist[[list]][item];
      if(!is.na(element)) {
        toSquare[list, element] = 1;
      }
    }
  } 
  
  return(toSquare);
}#listtosquare

#test calls
squaretolist(ms);
squaretothin(ms);
thintolist(mt, 4);
thintosquare(mt, 4);
listtothin(ml);
listtosquare(ml, 4);

